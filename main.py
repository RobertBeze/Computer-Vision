import cv2
import numpy as np
import utils
from brighten_image import *
import os
###########
path = "./grading"
path_answ = "./answers"
finals = "./final"
width = 700
height = 700
questions = 5
choices = 5
ans = []
JSON_output = list()
CSV_output = [["Email", "Nota"]]
#dsc1234.raw
###########



def get_correct_ans(path_answ):
    global ans, width, height, questions, choices
    files = os.listdir(path_answ)
    for name in files:
        if name.endswith("jpg") or name.endswith("png") or name.endswith("jpeg") or name.endswith("HEIC"):
            img = path_answ + "/" +name
            break

    img = cv2.imread(img)
    img = brigthen(img) if isDark(img) else img #daca imaginea este intunecata, lumineaz-o
    # schimb dimensiunea imaginii
    img = cv2.resize(img, (width, height))
    imgContours = img.copy()

    # convertesc in greyscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # adaug blur Gaussian asupra imaginii in greyscale
    # GaussianBlur reduce din noise-ul imaginii, ea devine astfel mai smooth
    # mai usor de prelucrat
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # img, kernel size, sigma

    # transform in Canny
    # folosesc canny pentru a detecta liniile din imagini
    # si mai tarziu sa caut contururile din imagine
    imgCanny = cv2.Canny(imgBlur, 10, 50)  # img, treshold1, treshold2

    # caut contururile imaginilor
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # desenez contururile
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # caut patrulatere in contours
    # ma intereseaza cele mai mari doua patrulatere
    lista_patrulatere = utils.rectContour(contours)
    gradeArea = utils.getCornerPoints(lista_patrulatere[0])  # cel mai mare, gradeArea
    gradePoints = utils.getCornerPoints(lista_patrulatere[1])  # al doilea cel mai mare, gradePointBox

    # afisez punctele din colturile cele mai mari doua patrulatere
    # daca sunt gasite
    if gradeArea.size != 0 and gradePoints.size != 0:
        imgGradeArea = img.copy()
        cv2.drawContours(imgGradeArea, gradeArea, -1, (0, 255, 0), 20)
        cv2.drawContours(imgGradeArea, gradePoints, -1, (255, 0, 0), 20)

        # reordonez punctele astfel incat sa fie in ordine
        # in functie de origine A->B, C->D
        #   C       D
        #
        #   A       B
        gradeArea = utils.reorder(gradeArea)
        gradePoints = utils.reorder(gradePoints)

        pt1 = np.float32(gradeArea)
        pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)  # matricea de transformare pentru
        # perspectiva de tip birdsEye(de sus)
        imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))

        ptG1 = np.float32(gradePoints)
        ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)  # matricea de transformare pentru
        # perspectiva de tip birdsEye(de sus)
        imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
        # cv2.imshow("Grade Points Area", imgGradeDisplay)

        # aplic un treshold pentru a gasii punctele bifate
        # pe baza treshold-ului gasesc puntele bifate si nebifate
        # cele bifate vor avea mai multi pixeli de culoare alba
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

        # pe patrulaterul cu threshold fac splitting pentru a obtine toata coloanele/randurile
        # pentru a obtine toate bulinele bifate/nebifate
        buline = utils.splitBoxes(imgThresh)
        # cv2.imshow("Test", buline[2]) Afisez bulina nr 3
        # print(cv2.countNonZero(buline[1]), cv2.countNonZero(buline[2]))
        # afisez nr de pixeli albi pe fiecare buline
        # utilizat pentru testare, normal vs bifat

        # am nevoie de un array de 5x5
        # pentru a retine valorile non zero din fiecare imagine
        myPixelValues = np.zeros((questions, choices))
        countC = 0
        countR = 0

        for image in buline:
            totalPixels = cv2.countNonZero(image)
            myPixelValues[countR][countC] = totalPixels
            countC = countC + 1
            if countC == choices:
                countC = 0
                countR = countR + 1
        #print(myPixelValues)  # testare array mypixelvalues

        # caut bulinele bifate din imagine
        # bulina bifata reprezinta maximul din fiecare row
        myIndex = []
        for x in range(0, questions):
            arr = myPixelValues[x]
            # aflu maxim din array
            # index_maximum = np.where(arr==np.amax(arr))
            index_maximum = np.where(arr > np.mean(arr))
            index_maximum = index_maximum[0].tolist()
            #print(index_maximum)
            myIndex.append(index_maximum)
        #print(myIndex)  # lista cu raspunsurile bifate
        ans = myIndex.copy()

get_correct_ans(path_answ)
print(ans)

def grade_teste(path):
    global ans, width, height, questions, choices

    files = os.listdir(path)
    for name in files:
        if name.endswith("jpg") or name.endswith("png") or name.endswith("jpeg") or name.endswith("HEIC"):
            img = path + "/" +name
            print(img)
            img = cv2.imread(img)
            img = brigthen(img) if isDark(img) else img #daca e intunecata, lumineaz-o
            #cv2.imshow("Test", img)

            # schimb dimensiunea imaginii
            img = cv2.resize(img, (width, height))
            imgContours = img.copy()

            # convertesc in greyscale
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # adaug blur Gaussian asupra imaginii in greyscale
            # GaussianBlur reduce din noise-ul imaginii, ea devine astfel mai smooth
            # mai usor de prelucrat
            imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # img, kernel size, sigma

            # transform in Canny
            # folosesc canny pentru a detecta liniile din imagini
            # si mai tarziu sa caut contururile din imagine
            imgCanny = cv2.Canny(imgBlur, 10, 50)  # img, treshold1, treshold2

            # caut contururile imaginilor
            contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # desenez contururile
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

            # caut patrulatere in contours
            # ma intereseaza cele mai mari doua patrulatere
            lista_patrulatere = utils.rectContour(contours)
            gradeArea = utils.getCornerPoints(lista_patrulatere[0])  # cel mai mare, gradeArea
            gradePoints = utils.getCornerPoints(lista_patrulatere[1])  # al doilea cel mai mare, gradePointBox

            # afisez punctele din colturile cele mai mari doua patrulatere
            # daca sunt gasite
            if gradeArea.size != 0 and gradePoints.size != 0:
                imgGradeArea = img.copy()
                cv2.drawContours(imgGradeArea, gradeArea, -1, (0, 255, 0), 20)
                cv2.drawContours(imgGradeArea, gradePoints, -1, (255, 0, 0), 20)

                # reordonez punctele astfel incat sa fie in ordine
                # in functie de origine A->B, C->D
                #   C       D
                #
                #   A       B
                gradeArea = utils.reorder(gradeArea)
                gradePoints = utils.reorder(gradePoints)

                pt1 = np.float32(gradeArea)
                pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                matrix = cv2.getPerspectiveTransform(pt1, pt2)  # matricea de transformare pentru
                # perspectiva de tip birdsEye(de sus)
                imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))

                ptG1 = np.float32(gradePoints)
                ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
                matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)  # matricea de transformare pentru
                # perspectiva de tip birdsEye(de sus)
                imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))
                # cv2.imshow("Grade Points Area", imgGradeDisplay)

                # aplic un treshold pentru a gasii punctele bifate
                # pe baza treshold-ului gasesc puntele bifate si nebifate
                # cele bifate vor avea mai multi pixeli de culoare alba
                imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
                imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

                # pe patrulaterul cu threshold fac splitting pentru a obtine toata coloanele/randurile
                # pentru a obtine toate bulinele bifate/nebifate
                buline = utils.splitBoxes(imgThresh)
                # cv2.imshow("Test", buline[2]) Afisez bulina nr 3
                # print(cv2.countNonZero(buline[1]), cv2.countNonZero(buline[2]))
                # afisez nr de pixeli albi pe fiecare buline
                # utilizat pentru testare, normal vs bifat

                # am nevoie de un array de 5x5
                # pentru a retine valorile non zero din fiecare imagine
                myPixelValues = np.zeros((questions, choices))
                countC = 0
                countR = 0

                for image in buline:
                    totalPixels = cv2.countNonZero(image)
                    myPixelValues[countR][countC] = totalPixels
                    countC = countC + 1
                    if countC == choices:
                        countC = 0
                        countR = countR + 1
                # print(myPixelValues) #testare array mypixelvalues

                # caut bulinele bifate din imagine
                # bulina bifata reprezinta maximul din fiecare row
                myIndex = []
                for x in range(0, questions):
                    arr = myPixelValues[x]
                    # aflu maxim din array
                    # index_maximum = np.where(arr==np.amax(arr))
                    index_maximum = np.where(arr > np.mean(arr))
                    index_maximum = index_maximum[0].tolist()
                    # print(index_maximum)
                    myIndex.append(index_maximum)
                # print(myIndex) #lista cu raspunsurile bifate

                grading = []
                for x in range(0, questions):
                    tmp = ans[x]
                    tmp2 = myIndex[x]

                    tmp_set = set(tmp)
                    tmp2_set = set(tmp2)
                    aux = tmp_set.intersection(tmp2_set)
                    if len(tmp_set) != len(tmp2_set):
                        grading.append(0)
                    elif tmp_set == tmp2_set:
                        grading.append(1)
                    else:
                        grading.append(0)
                print(grading)  # testare cate raspunsuri au fost bifate corect
                score = (sum(grading) / questions) * 100
                print(score)  # afisez punctajul final

                # afisare raspunsuri corecte/gresite
                imgResult = imgWarpColored.copy()
                imgResult = utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
                imRawDrawing = np.zeros_like(imgWarpColored)
                imRawDrawing = utils.showAnswers(imRawDrawing, myIndex, grading, ans, questions, choices)

                # reverse la perspectiva pe imrawdrawing
                invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
                imgInvWarp = cv2.warpPerspective(imRawDrawing, invMatrix, (width, height))

                # reverse perspective pe imrawgrade
                imgRawGrade = np.zeros_like(imgGradeDisplay)
              #  cv2.putText(imgRawGrade, str(int(score)) + "%", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                utils.draw_text(imgRawGrade, str(int(score)) + "%", cv2.FONT_HERSHEY_SIMPLEX, (60, 40), 3, 2, (0, 255, 0), (0, 0, 0))
             #   cv2.imshow("Grade", imgRawGrade)
              #  cv2.waitKey(0)
                invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
                imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (width, height))
                # cv2.imshow("Grade test", imgGradeDisplay)

                # combin imaginile, imginvwarp si originalul
                imgFinal = img.copy()
                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
                imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)
                #cv2.imshow("Final", imgFinal)
                global finals
                pth = finals + "/" + name
                cv2.imwrite(pth,imgFinal)
                tmp = pth.split('/')
                if name!='output.jpg':
                    JSON_output.append({name[:len(name)-4:]: str(int(score))})
                    CSV_output.append([name[:len(name)-4:], str(int(score))])
    from json import dump
    with open('results.json', 'w+') as f:
        dump(JSON_output, f, indent = 4)
    from csv import writer
    with open('results.csv', 'w+') as f:
        csv_writer = writer(f)
        csv_writer.writerows(CSV_output)

grade_teste(path)
