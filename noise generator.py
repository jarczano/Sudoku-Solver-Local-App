import cv2
import numpy as np

output_folder = "szumowe_obrazy/"

for i in range(1):
    # Generowanie obrazu szumu
    noise_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)

    # Zapisywanie obrazu do pliku
    filename = output_folder + f"szum_{i}.png"
    cv2.imshow('img', noise_image)
    cv2.waitKey(0)


    #cv2.imwrite(filename, noise_image)

    if (i + 1) % 1000 == 0:
        print(f"Utworzono {i + 1} obrazów szumu")

print("Generowanie obrazów szumu zakończone.")