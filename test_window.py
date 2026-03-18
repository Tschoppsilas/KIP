import os, sys
print("=== Umgebung ===")
print("DISPLAY       :", os.environ.get("DISPLAY", "NICHT GESETZT"))
print("WAYLAND       :", os.environ.get("WAYLAND_DISPLAY", "NICHT GESETZT"))
print("QT_PLATFORM   :", os.environ.get("QT_QPA_PLATFORM", "NICHT GESETZT"))
print("XAUTHORITY    :", os.environ.get("XAUTHORITY", "NICHT GESETZT"))
print()

import cv2, numpy as np
print("=== cv2 Window Test ===")
img = np.zeros((300, 600, 3), dtype=np.uint8)
cv2.putText(img, "Wenn du das siehst: Fenster OK!", (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.putText(img, "Druecke beliebige Taste zum Schliessen", (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

cv2.namedWindow("FENSTER_TEST", cv2.WINDOW_NORMAL)
vis = cv2.getWindowProperty("FENSTER_TEST", cv2.WND_PROP_VISIBLE)
print("Fenster visible:", vis)

if vis >= 0:
    cv2.imshow("FENSTER_TEST", img)
    key = cv2.waitKey(0)
    print("Taste gedrueckt:", key)
    cv2.destroyAllWindows()
    print("Fenster OK - Test bestanden!")
else:
    print("FEHLER: Fenster konnte nicht erstellt werden")
    sys.exit(1)
