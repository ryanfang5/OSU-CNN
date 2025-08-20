# OSU CNN
This project explores using a neural network to play the rhythm game osu! by predicting (x, y) mouse coordinates on the screen based on real-time screenshots of the game. Currently, the network only moves the mouse to the objects on the screen but does not yet support clicking on the beat.
The network is trained on ~50,000 samples of data based on my own gameplay.

**Link to demonstration:** 

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/jHhrI1sotU0/0.jpg)](https://www.youtube.com/watch?v=jHhrI1sotU0)

## How It's Made:

The first step was to obtain a live, high-performance video stream from the game window. I opted to use the pygetwindow library to calculate both the size and location of the game window, allowing screenshots obtained using dxcam to contain only the game window rather than the entire screen. Pairs of 
images and normalized mouse coordinates were then saved and stored in numpy arrays to use as training data. The next step was building and training the neural network. After creating a custom pytorch dataset, I based the model on a modified version of ResNet-50, with its final fully-connected layer replaced to output two continuous variables representing the (x, y) coordinates. Training was done using a simple version of early stopping and monitored through Tensorboard. Finally, the mouse library was used to move the cursor to the coordinates predicted by the model based on the image inputted once training was complete.

## Optimizations

I had initially used the win32api and pyautogui libraries to take screenshots of the window and move the cursor. However, replacing these with dxcam and mouse libraries resulted in a 5-10 times increase in FPS. Furthermore, storing normalized mouse coordinates from 0-1 based on the location of the game window instead of the monitor allowed
screenshots to be taken in a lower initial resolution, doubling performance.

