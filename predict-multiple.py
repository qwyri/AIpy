from keras.models import load_model
from keras.utils import load_img
import numpy as np
import datetime;

modelnames = ["test.h5","genshin_v1.h5","genshin_v4.h5", "ai_genshin.h5", "aiv2_genshin.h5"]
imagenames = ["ac8b4c22911f70c3b814f91df7f122e2.jpg","1892566ff908fad9d014d41b08d2d3dd.jpg"]
humanpredictions = ["AI","NOT AI"]

f = open("result.txt", "w")
f.write(f"\n\n- LOG | {datetime.datetime.now()} |")
f.close()

for i in range(len(modelnames)):
    f = open("result.txt", "a")
    f.write(f"\n\n- LOG | {datetime.datetime.now()} |")
    f.close()
    for j in range(len(imagenames)):
        model = load_model(rf'{modelnames[i]}')

        image = load_img(f'{imagenames[j]}', target_size=(224, 224))
        img = np.array(image)
        img = img / 255.0
        img = img.reshape(1,224,224,3)
        label = model.predict(img)

        if np.where(label[0][0] > 0.5, 1,0) == 1:
            lm = "Not AI"
        else:
            lm = "AI"

        predclass = f"Predicted Class (0 - Ai , 1 - Not): {label[0][0]}"
        confrate = f"Confidence Rate (0% - Ai Generated, 100% - Original): {round((label[0][0])*100)}%"
        humanvsh5 = f"Human vs AI 'AI Detection' (HUMAN:{humanpredictions[j]}) & (AI:{lm})"

        f = open("result.txt", "a")
        f.write(f"\n\n{modelnames[i]}\n{imagenames[j]} | {humanvsh5}\n{predclass}\n{confrate}")
        f.close()