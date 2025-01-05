import cv2
import numpy as np
from collections import deque


class BackgroundExtraction:
    def __init__(self, width, height, scale, maxlen=10):
        self.maxlen = maxlen
        self.scale = scale
        self.width = width // scale
        self.height = height // scale
        self.buffer = deque(maxlen=maxlen)
        self.background = None

    def calculate_background(self):
        self.background = np.zeros((self.height, self.width), dtype='float32')
        for item in self.buffer:
            self.background += item
        self.background /= len(self.buffer)

    def update_background(self, old_frame, new_frame):
        self.background -= old_frame / self.maxlen
        self.background += new_frame / self.maxlen

    def update_frame(self, frame):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(frame)
            self.calculate_background()
        else:
            old_frame = self.buffer.popleft()
            self.buffer.append(frame)
            self.update_background(old_frame, frame)

    def get_background(self):
        return self.background.astype('uint8')

    def apply(self, frame):
        down_scale = cv2.resize(frame, (self.width, self.height))
        gray = cv2.cvtColor(down_scale, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        self.update_frame(gray)
        abs_diff = cv2.absdiff(self.get_background(), gray)
        _, ad_mask = cv2.threshold(abs_diff, 15, 255, cv2.THRESH_BINARY)
        return cv2.resize(ad_mask, (self.width * self.scale, self.height * self.scale))


class PlayGame:
    def __init__(self, width, height, size=50):
        self.width = width
        self.height = height
        self.size = size
        self.logo = cv2.imread("logo.png")
        if self.logo is None:
            raise FileNotFoundError("Error: The image file logo.png was not found.")
        self.logo = cv2.resize(self.logo, (self.size, self.size))
        gray = cv2.cvtColor(self.logo, cv2.COLOR_BGR2GRAY)
        self.mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
        self.reset_game()

    def reset_game(self):
        self.x = np.random.randint(0, self.width - self.size)
        self.y = 0
        self.speed = 15
        self.score = 0
        self.lives = 5
        self.game_over = False

    def update_frame(self, frame):
        roi = frame[self.y:self.y + self.size, self.x:self.x + self.size]
        roi[np.where(self.mask)] = 0
        roi += self.logo

    def show_game_over(self, frame):
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Add game over text
        text = "GAME OVER"
        score_text = f"Final Score: {self.score}"
        restart_text = "Press 'R' to Restart"
        quit_text = "Press 'Q' to Quit"

        # Calculate text sizes and positions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 2

        # Position text in center
        textsize = cv2.getTextSize(text, font, font_scale, thickness)[0]
        textX = (self.width - textsize[0]) // 2
        textY = (self.height - textsize[1]) // 2

        # Draw all text
        cv2.putText(frame, text, (textX, textY), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, score_text, (textX, textY + 50), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, restart_text, (textX, textY + 100), font, 1, (255, 255, 255), thickness)
        cv2.putText(frame, quit_text, (textX, textY + 140), font, 1, (255, 255, 255), thickness)

        return frame

    def update_position(self, fg_mask):
        if self.game_over:
            return False

        self.y += self.speed
        if self.y + self.size >= self.height:
            self.lives -= 1
            self.y = 0
            self.speed = np.random.randint(10, 15)
            self.x = np.random.randint(0, self.width - self.size)
            if self.lives <= 0:
                self.game_over = True
            return False

        roi = fg_mask[self.y:self.y + self.size, self.x:self.x + self.size]
        check = np.any(roi[np.where(self.mask)])
        if check:
            self.score += 1
            self.y = 0
            self.speed = np.random.randint(10, 20)
            self.x = np.random.randint(0, self.width - self.size)
        return check


width = 640
height = 480
scale = 2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

bg_buffer = BackgroundExtraction(width, height, scale, maxlen=5)
game = PlayGame(width, height)

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)

    fg_mask = bg_buffer.apply(frame)
    collision = game.update_position(fg_mask)

    if not game.game_over:
        game.update_frame(frame)
        if collision:
            frame[::2] = 255

        # Display score and lives
        score_text = f"Score: {game.score}"
        lives_text = f"Lives: {game.lives}"
        cv2.putText(frame, score_text, (30, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 3)
        cv2.putText(frame, lives_text, (30, 90), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 3)
    else:
        frame = game.show_game_over(frame)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r') and game.game_over:
        game.reset_game()

cap.release()
cv2.destroyAllWindows()
