import codecs
import csv
import json
import os
import pickle
import subprocess
import sys

import gtabview
import h5py
import keras.backend as K
import matplotlib as mpl
import pandas
import scipy.io as sio
import tensorflow as tf
import yaml
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAbstractItemView, QFileDialog
from keras.models import load_model
from keras.utils.vis_utils import plot_model, model_to_dot
from matplotlib.path import Path

from configGUI.canvas import Canvas
from configGUI.labelDialog import LabelDialog
from configGUI.labelTable import LabelTable

mpl.use('Qt5Agg')
from matplotlib import path, colors
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Ellipse, PathPatch

from config.DatabaseInfo import DatabaseInfo

from configGUI import network_visualization
from configGUI.Grey_window import grey_window
from configGUI.Patches_window import Patches_window
from configGUI.framework import Ui_MainWindow
from configGUI.loadf2 import *
from configGUI.matplotlibwidget import MyMplCanvas
from configGUI.Unpatch import UnpatchType, UnpatchArte
from configGUI.Unpatch_eight import UnpatchArte8
from configGUI.Unpatch_two import fUnpatch2D
from configGUI.activescene import Activescene
from configGUI.activeview import Activeview
from configGUI.cPre_window import cPre_window
from configGUI.label_window import Label_window
from configGUI.loadf import loadImage
from DLart.dlart import DeepLearningArtApp

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):

    update_data = QtCore.pyqtSignal(list)
    gray_data = QtCore.pyqtSignal(list)
    new_page = QtCore.pyqtSignal()
    update_data2 = QtCore.pyqtSignal(list)
    gray_data2 = QtCore.pyqtSignal(list)
    new_page2 = QtCore.pyqtSignal()
    update_data3 = QtCore.pyqtSignal(list)
    gray_data3 = QtCore.pyqtSignal(list)
    new_page3 = QtCore.pyqtSignal()

    def __init__(self):
        super(MyApp, self).__init__()
        self.setupUi(self)

        self.scrollAreaWidgetContents1 = QtWidgets.QWidget()
        self.maingrids1 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents1)
        self.scrollArea1.setWidget(self.scrollAreaWidgetContents1)
        # self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrollAreaWidgetContents2 = QtWidgets.QWidget()
        self.maingrids2 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents2)
        self.scrollArea2.setWidget(self.scrollAreaWidgetContents2)
        self.stackedWidget.setCurrentIndex(0)
        # Main widgets and related state.

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''
        self.listOfLabel = []

        self.vision = 2
        self.gridson = False
        self.gridsnr = 2
        self.voxel_ndarray = []
        self.i = -1
        self.linked = False
        self.imageshow = True # 1-On/0-Off
        self.viewAll = True

        with open('configGUI/editlabel.json', 'r') as json_data:
            self.infos = json.load(json_data)
            self.labelnames = self.infos['names']
            self.labelcolor = self.infos['colors']
            self.pathROI = self.infos['path'][0]

        global pathlist, list1, shapelist, pnamelist, empty1, cmap1, cmap3, hmap1, hmap2, vtr1,  \
            vtr3, problist, hatchlist, correslist, cnrlist, indlist, ind2list, ind3list
        pathlist = []
        list1 = []
        shapelist = []
        pnamelist = []
        empty1 = []
        indlist = []
        ind2list = []
        ind3list = []

        with open('configGUI/colors0.json', 'r') as json_data:
            self.dcolors = json.load(json_data)
            cmap1 = self.dcolors['class2']['colors']
            cmap1 = mpl.colors.ListedColormap(cmap1)
            cmap3 = self.dcolors['class11']['colors']
            cmap3 = mpl.colors.ListedColormap(cmap3)
            hmap1 = self.dcolors['class8']['hatches']
            hmap2 = self.dcolors['class11']['hatches']
            vtr1 = self.dcolors['class2']['trans'][0]
            vtr3 = self.dcolors['class11']['trans'][0]

        problist = []
        hatchlist = []
        correslist = []
        cnrlist = []

        self.newfig = plt.figure(figsize=(8, 6)) # 3
        self.newfig.set_facecolor("black")
        self.newax = self.newfig.add_subplot(111)
        self.newax.axis('off')
        self.pltc = None
        self.newcanvas = FigureCanvas(self.newfig)
        self.keylist = []
        self.mrinmain = None
        self.labelimage = False

        self.newfig2 = plt.figure(figsize=(8, 6))
        self.newfig2.set_facecolor("black")
        self.newax2 = self.newfig2.add_subplot(111)
        self.newax2.axis('off')
        self.pltc2 = None
        self.newcanvas2 = FigureCanvas(self.newfig2)  # must be defined because of selector next
        self.shownLabels2 = []
        self.keylist2 = []  # locate the key in combobox
        self.limage2 = None

        self.newfig3 = plt.figure(figsize=(8, 6))
        self.newfig3.set_facecolor("black")
        self.newax3 = self.newfig3.add_subplot(111)
        self.newax3.axis('off')
        self.pltc3 = None
        self.newcanvas3 = FigureCanvas(self.newfig3)  # must be defined because of selector next
        self.shownLabels3 = []
        self.keylist3 = []  # locate the key in combobox
        self.limage3 = None

        self.graylabel = QtWidgets.QLabel()
        self.slicelabel = QtWidgets.QLabel()
        self.zoomlabel = QtWidgets.QLabel()
        self.graylabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.graylabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slicelabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.slicelabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zoomlabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.zoomlabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.seditgray = QtWidgets.QPushButton()
        self.sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.sizePolicy.setHorizontalStretch(0)
        self.sizePolicy.setVerticalStretch(0)
        self.sizePolicy.setHeightForWidth(self.seditgray.sizePolicy().hasHeightForWidth())
        self.seditgray.setSizePolicy(self.sizePolicy)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/Icons/edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.seditgray.setText("")
        self.seditgray.setIcon(icon3)
        self.maingrids2.addWidget(self.slicelabel, 0, 0, 1, 1)
        self.maingrids2.addWidget(self.zoomlabel, 0, 1, 1, 1)
        self.maingrids2.addWidget(self.graylabel, 0, 2, 1, 1)
        self.maingrids2.addWidget(self.seditgray, 0, 3, 1, 1)
        self.viewLabel = Activeview()
        self.sceneLabel = Activescene()
        self.sceneLabel.addWidget(self.newcanvas)
        self.viewLabel.setScene(self.sceneLabel)
        self.maingrids2.addWidget(self.viewLabel, 1, 0, 1, 4)
        self.graylabel2 = QtWidgets.QLabel()
        self.slicelabel2 = QtWidgets.QLabel()
        self.zoomlabel2 = QtWidgets.QLabel()
        self.graylabel2.setFrameShape(QtWidgets.QFrame.Panel)
        self.graylabel2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slicelabel2.setFrameShape(QtWidgets.QFrame.Panel)
        self.slicelabel2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zoomlabel2.setFrameShape(QtWidgets.QFrame.Panel)
        self.zoomlabel2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.seditgray2 = QtWidgets.QPushButton()
        self.sizePolicy.setHeightForWidth(self.seditgray2.sizePolicy().hasHeightForWidth())
        self.seditgray2.setSizePolicy(self.sizePolicy)
        self.seditgray2.setText("")
        self.seditgray2.setIcon(icon3)
        self.maingrids2.addWidget(self.slicelabel2, 0, 4, 1, 1)
        self.maingrids2.addWidget(self.zoomlabel2, 0, 5, 1, 1)
        self.maingrids2.addWidget(self.graylabel2, 0, 6, 1, 1)
        self.maingrids2.addWidget(self.seditgray2, 0, 7, 1, 1)
        self.viewLabel2 = Activeview()
        self.sceneLabel2 = Activescene()
        self.sceneLabel2.addWidget(self.newcanvas2)
        self.viewLabel2.setScene(self.sceneLabel2)
        self.maingrids2.addWidget(self.viewLabel2, 1, 4, 1, 4)

        self.graylabel3 = QtWidgets.QLabel()
        self.slicelabel3 = QtWidgets.QLabel()
        self.zoomlabel3 = QtWidgets.QLabel()
        self.graylabel3.setFrameShape(QtWidgets.QFrame.Panel)
        self.graylabel3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slicelabel3.setFrameShape(QtWidgets.QFrame.Panel)
        self.slicelabel3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zoomlabel3.setFrameShape(QtWidgets.QFrame.Panel)
        self.zoomlabel3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.seditgray3 = QtWidgets.QPushButton()
        self.sizePolicy.setHeightForWidth(self.seditgray3.sizePolicy().hasHeightForWidth())
        self.seditgray3.setSizePolicy(self.sizePolicy)
        self.seditgray3.setText("")
        self.seditgray3.setIcon(icon3)
        self.maingrids2.addWidget(self.slicelabel3, 0, 8, 1, 1)
        self.maingrids2.addWidget(self.zoomlabel3, 0, 9, 1, 1)
        self.maingrids2.addWidget(self.graylabel3, 0, 10, 1, 1)
        self.maingrids2.addWidget(self.seditgray3, 0, 11, 1, 1)
        self.viewLabel3 = Activeview()
        self.sceneLabel3 = Activescene()
        self.sceneLabel3.addWidget(self.newcanvas3)
        self.viewLabel3.setScene(self.sceneLabel3)
        self.maingrids2.addWidget(self.viewLabel3, 1, 8, 1, 4)

        self.openfile.clicked.connect(self.loadMR)
        self.bdswitch.clicked.connect(self.switchview)
        self.bgrids.clicked.connect(self.set_layout)
        self.bpatch.clicked.connect(self.load_patch)
        self.bsetcolor.clicked.connect(self.set_color)

        self.bselectoron.clicked.connect(self.selector_mode)
        # self.bchoosemark.clicked.connect(self.chooseMark)
        self.imageShow.clicked.connect(self.image_on_off)
        self.roiButton.clicked.connect(self.select_roi_OnOff)
        self.inspectorButton.clicked.connect(self.mouse_tracking)

        self.brectangle.setDisabled(True)
        self.bellipse.setDisabled(True)
        self.blasso.setDisabled(True)
        # self.bchoosemark.setDisabled(True)
        self.bnoselect.setDisabled(True)
        self.cursorCross.setDisabled(True)
        self.bnoselect.setChecked(True)
        self.deleteButton.setDisabled(True)
        self.openlabelButton.setDisabled(True)
        self.useDefaultLabelCheckbox.setDisabled(True)
        self.defaultLabelTextLine.setDisabled(True)
        self.labellistBox.setDisabled(True)
        self.infoButton.setDisabled(True)
        self.viewallButton.setDisabled(True)
        self.bnoselect.toggled.connect(lambda:self.marking_shape(0))
        self.brectangle.toggled.connect(lambda:self.marking_shape(1))
        self.bellipse.toggled.connect(lambda:self.marking_shape(2))
        self.blasso.toggled.connect(lambda:self.marking_shape(3))

        self.brectangle.toggled.connect(lambda:self.stop_view(1))
        self.bellipse.toggled.connect(lambda:self.stop_view(1))
        self.blasso.toggled.connect(lambda: self.stop_view(1))
        self.bnoselect.toggled.connect(lambda: self.stop_view(0))

        self.view_group = None

        self.selectoron = False
        self.x_clicked = None
        self.y_clicked = None
        self.mouse_second_clicked = False

        self.actionOpen_file.triggered.connect(self.loadMR)
        self.actionSave.triggered.connect(self.save_current)
        self.actionLoad.triggered.connect(self.load_old)
        self.actionColor.triggered.connect(self.default_color)
        self.actionLabels.triggered.connect(self.show_label_info)
        self.cursorCross.setChecked(False)
        self.cursorCross.toggled.connect(self.handle_crossline)

        self.ind = 0
        self.ind2 = 0
        self.ind3 = 0
        self.slices = 0
        self.slices2 = 0
        self.slices3 = 0
        self.newcanvas.mpl_connect('scroll_event', self.newonscroll)
        self.newcanvas.mpl_connect('button_press_event', self.mouse_clicked)
        self.newcanvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.newcanvas.mpl_connect('button_release_event', self.mouse_release)

        self.newcanvas2.mpl_connect('scroll_event', self.newonscrol2)
        self.newcanvas2.mpl_connect('button_press_event', self.mouse_clicked2)
        self.newcanvas2.mpl_connect('motion_notify_event', self.mouse_move2)
        self.newcanvas2.mpl_connect('button_release_event', self.mouse_release2)

        self.newcanvas3.mpl_connect('scroll_event', self.newonscrol3)
        self.newcanvas3.mpl_connect('button_press_event', self.mouse_clicked3)
        self.newcanvas3.mpl_connect('motion_notify_event', self.mouse_move3)
        self.newcanvas3.mpl_connect('button_release_event', self.mouse_release3)

        self.labelHist = []
        self.loadPredefinedClasses('configGUI/predefined_classes.txt')
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.labelfile = 'Markings/marking_records.csv'
        self.deleteButton.clicked.connect(self.delete_labelItem)
        self.openlabelButton.clicked.connect(self.open_labelFile)
        self.viewallButton.clicked.connect(self.viewAllLabel_on_off)
        self.infoButton.clicked.connect(self.show_label_info)

############ second tab from Yannick
        # initialize DeepLearningArt Application
        self.deepLearningArtApp = DeepLearningArtApp()
        self.deepLearningArtApp.setGUIHandle(self)

        # initialize TreeView Database
        self.manageTreeView()

        # intialize TreeView Datasets
        self.manageTreeViewDatasets()

        # initiliaze patch output path
        self.Label_OutputPathPatching.setText(self.deepLearningArtApp.getOutputPathForPatching())

        # initialize markings path
        self.Label_MarkingsPath.setText(self.deepLearningArtApp.getMarkingsPath())

        # initialize learning output path
        self.Label_LearningOutputPath.setText(self.deepLearningArtApp.getLearningOutputPath())

        # initialize patching mode
        self.ComboBox_Patching.setCurrentIndex(1)

        # initialize store mode
        self.ComboBox_StoreOptions.setCurrentIndex(0)

        # initialize splitting mode
        self.ComboBox_splittingMode.setCurrentIndex(DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING)
        self.Label_SplittingParams.setText("using Test/Train="
                                           + str(self.deepLearningArtApp.getTrainTestDatasetRatio())
                                           + " and Valid/Train=" + str(
            self.deepLearningArtApp.getTrainValidationRatio()))

        # initialize combox box for DNN selection
        self.ComboBox_DNNs.addItem("Select Deep Neural Network Model...")
        self.ComboBox_DNNs.addItems(DeepLearningArtApp.deepNeuralNetworks.keys())
        self.ComboBox_DNNs.setCurrentIndex(1)
        self.deepLearningArtApp.setNeuralNetworkModel(self.ComboBox_DNNs.currentText())

        # initialize check boxes for used classes
        self.CheckBox_Artifacts.setChecked(self.deepLearningArtApp.getUsingArtifacts())
        self.CheckBox_BodyRegion.setChecked(self.deepLearningArtApp.getUsingBodyRegions())
        self.CheckBox_TWeighting.setChecked(self.deepLearningArtApp.getUsingTWeighting())

        # initilize training parameters
        self.DoubleSpinBox_WeightDecay.setValue(self.deepLearningArtApp.getWeightDecay())
        self.DoubleSpinBox_Momentum.setValue(self.deepLearningArtApp.getMomentum())
        self.CheckBox_Nesterov.setChecked(self.deepLearningArtApp.getNesterovEnabled())
        self.CheckBox_DataAugmentation.setChecked(self.deepLearningArtApp.getDataAugmentationEnabled())
        self.CheckBox_DataAug_horizontalFlip.setChecked(self.deepLearningArtApp.getHorizontalFlip())
        self.CheckBox_DataAug_verticalFlip.setChecked(self.deepLearningArtApp.getVerticalFlip())
        self.CheckBox_DataAug_Rotation.setChecked(False if self.deepLearningArtApp.getRotation() == 0 else True)
        self.CheckBox_DataAug_zcaWeighting.setChecked(self.deepLearningArtApp.getZCA_Whitening())
        self.CheckBox_DataAug_HeightShift.setChecked(False if self.deepLearningArtApp.getHeightShift() == 0 else True)
        self.CheckBox_DataAug_WidthShift.setChecked(False if self.deepLearningArtApp.getWidthShift() == 0 else True)
        self.CheckBox_DataAug_Zoom.setChecked(False if self.deepLearningArtApp.getZoom() == 0 else True)
        self.check_dataAugmentation_enabled()

        # Signals and Slots

        # select database button clicked
        self.Button_DB.clicked.connect(self.button_DB_clicked)
        # self.Button_DB.clicked.connect(self.button_DB_clicked)

        # output path button for patching clicked
        self.Button_OutputPathPatching.clicked.connect(self.button_outputPatching_clicked)

        # TreeWidgets
        self.TreeWidget_Patients.clicked.connect(self.getSelectedPatients)
        self.TreeWidget_Datasets.clicked.connect(self.getSelectedDatasets)

        # Patching button
        self.Button_Patching.clicked.connect(self.button_patching_clicked)

        # mask marking path button clicekd
        self.Button_MarkingsPath.clicked.connect(self.button_markingsPath_clicked)

        # combo box splitting mode is changed
        self.ComboBox_splittingMode.currentIndexChanged.connect(self.splittingMode_changed)

        # "use current data" button clicked
        self.Button_useCurrentData.clicked.connect(self.button_useCurrentData_clicked)

        # select dataset is clicked
        self.Button_selectDataset.clicked.connect(self.button_selectDataset_clicked)

        # learning output path button clicked
        self.Button_LearningOutputPath.clicked.connect(self.button_learningOutputPath_clicked)

        # train button clicked
        self.Button_train.clicked.connect(self.button_train_clicked)

        # combobox dnns
        self.ComboBox_DNNs.currentIndexChanged.connect(self.selectedDNN_changed)

        # data augmentation enbaled changed
        self.CheckBox_DataAugmentation.stateChanged.connect(self.check_dataAugmentation_enabled)

##################################3
        self.matplotlibwidget_static.show()
        # self.matplotlibwidget_static_2.hide()
        self.scrollArea.show()
        self.horizontalSliderPatch.hide()
        self.horizontalSliderSlice.hide()

        self.labelPatch.hide()
        self.labelSlice.hide()
        # self.horizontalSliderSS.hide()

        self.lcdNumberPatch.hide()
        self.lcdNumberSlice.hide()
        # self.lcdNumberSS.hide()

        self.radioButton_3.hide()
        self.radioButton_4.hide()

        self.resetW=False
        self.resetF = False
        self.resetS = False

        self.twoInput=False
        self.chosenLayerName = []
        # the slider's value is the chosen patch's number
        self.chosenWeightNumber =1
        self.chosenWeightSliceNumber=1
        self.chosenPatchNumber = 1
        self.chosenPatchSliceNumber =1
        self.chosenSSNumber = 1
        self.openfile_name=''
        self.inputData_name=''
        self.inputData={}
        self.inputalpha = '0.19'
        self.inputGamma = '0.0000001'

        self.layer_index_name = {}
        self.model={}
        self.qList=[]
        self.totalWeights=0
        self.totalWeightsSlices =0
        self.totalPatches=0
        self.totalPatchesSlices =0
        self.totalSS=0

        self.modelDimension= ''
        self.modelName=''
        self.modelInput={}
        self.modelInput2={}
        self.ssResult={}
        self.activations = {}
        self.act = {}
        self.layers_by_depth={}
        self.weights ={}
        self.w={}
        self.LayerWeights = {}
        self.subset_selection = {}
        self.subset_selection_2 = {}
        self.radioButtonValue=[]
        self.listView.clicked.connect(self.clickList)

        self.W_F=''

        # slider of the weight and feature
        self.horizontalSliderPatch.sliderReleased.connect(self.sliderValue)
        self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)

        self.horizontalSliderSlice.sliderReleased.connect(self.sliderValue)
        self.horizontalSliderSlice.valueChanged.connect(self.lcdNumberSlice.display)

        # self.matplotlibwidget_static.mpl.wheel_scroll_W_signal.connect(self.wheelScrollW)
#        self.matplotlibwidget_static.mpl.wheel_scroll_signal.connect(self.wheelScroll)
        # self.matplotlibwidget_static.mpl.wheel_scroll_3D_signal.connect(self.wheelScroll)
        # self.matplotlibwidget_static.mpl.wheel_scroll_SS_signal.connect(self.wheelScrollSS)

        self.lineEdit.textChanged[str].connect(self.textChangeAlpha)
        self.lineEdit_2.textChanged[str].connect(self.textChangeGamma)


    def on_chooseModelFile_clicked(self):
        self.openfile_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose the file', '.', 'All files')[0]
        if len(self.openfile_name) == 0:
            pass
        else:
            self.horizontalSliderPatch.hide()
            self.horizontalSliderSlice.hide()
            self.labelPatch.hide()
            self.labelSlice.hide()
            self.lcdNumberSlice.hide()
            self.lcdNumberPatch.hide()
            self.matplotlibwidget_static.mpl.fig.clf()

            self.model = load_model(self.openfile_name)
            print()

    def switchview(self):
        if self.vision == 2:
            self.vision = 3
            self.visionlabel.setText('3D')
            self.columnbox.setCurrentIndex(2)
            self.columnlabel.setDisabled(True)
            self.columnbox.setDisabled(True)
            self.linebox.addItem("4")
            self.linebox.addItem("5")
            self.linebox.addItem("6")
            self.linebox.addItem("7")
            self.linebox.addItem("8")
            self.linebox.addItem("9")
        else:
            self.vision = 2
            self.visionlabel.setText('2D')
            self.columnbox.setCurrentIndex(0)
            self.linebox.removeItem(8)
            self.linebox.removeItem(7)
            self.linebox.removeItem(6)
            self.linebox.removeItem(5)
            self.linebox.removeItem(4)
            self.linebox.removeItem(3)
            self.columnlabel.setDisabled(False)
            self.columnbox.setDisabled(False)

    def image_on_off(self):

        if self.imageshow == True:
            self.imageshow = False
            self.showLabel.setText('OFF')
            icon12off = QtGui.QIcon()
            icon12off.addPixmap(QtGui.QPixmap(":/icons/Icons/eye off.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.imageShow.setIcon(icon12off)
            self.save_current()
            self.clearall()

        else:
            self.imageshow = True
            self.showLabel.setText('ON')
            icon12on = QtGui.QIcon()
            icon12on.addPixmap(QtGui.QPixmap(":/icons/Icons/eye on.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.imageShow.setIcon(icon12on)
            self.load_old()

    def viewAllLabel_on_off(self):

        if self.viewAll == True:
            self.viewAll = False
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/icons/Icons/eye off.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.viewallButton.setIcon(icon)
            self.labelList.view_all(True)

        else:
            self.viewAll = True
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/icons/Icons/eye on.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.viewallButton.setIcon(icon)
            self.labelList.view_all(False)

    def clearall(self):
        if self.gridsnr == 2:
            for i in reversed(range(self.maingrids1.count())):
                self.maingrids1.itemAt(i).clearWidgets()
                for j in reversed(range(self.maingrids1.itemAt(i).secondline.count())):
                    self.maingrids1.itemAt(i).secondline.itemAt(j).widget().setParent(None)
                self.maingrids1.removeItem(self.maingrids1.itemAt(i)) # invisible
        else:
            for i in reversed(range(self.maingrids1.count())):
                self.maingrids1.itemAt(i).clearWidgets()
                for j in reversed(range(self.maingrids1.itemAt(i).gridLayout_1.count())):
                    self.maingrids1.itemAt(i).gridLayout_1.itemAt(j).widget().setParent(None)
                for j in reversed(range(self.maingrids1.itemAt(i).gridLayout_2.count())):
                    self.maingrids1.itemAt(i).gridLayout_2.itemAt(j).widget().setParent(None)
                for j in reversed(range(self.maingrids1.itemAt(i).gridLayout_3.count())):
                    self.maingrids1.itemAt(i).gridLayout_3.itemAt(j).widget().setParent(None)
                for j in reversed(range(self.maingrids1.itemAt(i).gridLayout_4.count())):
                    self.maingrids1.itemAt(i).gridLayout_4.itemAt(j).widget().setParent(None)
                self.maingrids1.removeItem(self.maingrids1.itemAt(i))

    def save_current(self):
        if self.gridson:
            with open('configGUI/lastWorkspace.json', 'r') as json_data:
                lastState = json.load(json_data)
                lastState['mode'] = self.gridsnr ###
                if self.gridsnr == 2:
                    lastState['layout'][0] = self.layoutlines
                    lastState['layout'][1] = self.layoutcolumns
                else:
                    lastState['layout'][0] = self.layout3D

                global pathlist, list1, pnamelist, problist, hatchlist, correslist, cnrlist, shapelist, indlist, ind2list, ind3list
                # shapelist = (list(shapelist)).tolist()
                # shapelist = pd.Series(shapelist).to_json(orient='values')
                lastState['Shape'] = shapelist
                lastState['Pathes'] = pathlist
                lastState['NResults'] = pnamelist
                lastState['NrClass'] = cnrlist
                lastState['Corres'] = correslist
                lastState['Index'] = indlist
                lastState['Index2'] = ind2list
                lastState['Index3'] = ind3list

            with open('configGUI/lastWorkspace.json', 'w') as json_data:
                json_data.write(json.dumps(lastState))
                if self.imageshow == True:
                    QtWidgets.QMessageBox.information(self,
                                                          "Warning",
                                                          "Results have been saved",
                                                          QtWidgets.QMessageBox.Ok)

                else:
                    pass

            listA = open('config/dump1.txt', 'wb')
            pickle.dump(list1, listA)
            listA.close()
            listB = open('config/dump2.txt', 'wb')
            pickle.dump(problist, listB)
            listB.close()
            listC = open('config/dump3.txt', 'wb')
            pickle.dump(hatchlist, listC)
            listC.close()

            if self.selectoron:
                self.save_label()

    def load_old(self):
        self.clearall()
        global pathlist, list1, pnamelist, problist, hatchlist, correslist, cnrlist, shapelist, indlist, ind2list, ind3list

        with open('configGUI/lastWorkspace.json', 'r') as json_data:
            lastState = json.load(json_data)
            # list1 = lastState['listA']
            # problist = lastState['Probs']
            # hatchlist = lastState['Hatches']
            gridsnr = lastState['mode']  ##
            shapelist = lastState['Shape']
            pathlist = lastState['Pathes']
            pnamelist = lastState['NResults']
            cnrlist = lastState['NrClass']
            correslist = lastState['Corres']
            indlist = lastState['Index']
            ind2list = lastState['Index2']
            ind3list = lastState['Index3']

            if gridsnr == 2:
                if self.vision == 3:
                    self.switchview() # back to 2
                self.layoutlines = lastState['layout'][0]
                self.layoutcolumns = lastState['layout'][1]
                self.linebox.setCurrentIndex(self.layoutlines - 1)
                self.columnbox.setCurrentIndex(self.layoutcolumns - 1)
            else:
                if self.vision == 2:
                    self.switchview()
                self.layout3D = lastState['layout'][0]
                self.linebox.setCurrentIndex(self.layout3D - 1)

        listA = open('config/dump1.txt', 'rb')
        list1 = pickle.load(listA)
        listA.close()
        listB = open('config/dump2.txt', 'rb')
        problist = pickle.load(listB)
        listB.close()
        listC = open('config/dump3.txt', 'rb')
        hatchlist = pickle.load(listC)
        listC.close()

        self.set_layout()

    def set_layout(self):
        self.gridson = True
        if self.vision == 2:
            self.clearall()
            self.gridsnr = 2
            self.layoutlines = self.linebox.currentIndex() + 1
            self.layoutcolumns = self.columnbox.currentIndex() + 1
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    blocklayout = Viewgroup()
                    self.view_group = blocklayout
                    for dpath in pathlist:
                        blocklayout.addPathd(dpath, 0)
                    for cpath in pnamelist:
                        blocklayout.addPathre(cpath)
                    blocklayout.in_link.connect(self.link_mode)     # initial connection of in_link
                    self.maingrids1.addLayout(blocklayout, i, j)
            if pathlist:
                n = 0
                for i in range(self.layoutlines):
                    for j in range(self.layoutcolumns):
                        if n < len(pathlist):
                            self.maingrids1.itemAtPosition(i, j).pathbox.setCurrentIndex(n+1)
                            n+=1
                        else:
                            break
        else:
            self.clearall()
            self.gridsnr = 3
            self.layout3D = self.linebox.currentIndex() + 1
            for i in range(self.layout3D):
                blockline = Viewline()
                for dpath in pathlist:
                    blockline.addPathim(dpath)
                for cpath in pnamelist:
                    blockline.addPathre(cpath)
                blockline.in_link.connect(self.link_mode) # 3d initial
                self.maingrids1.addLayout(blockline, i, 0)
            if pathlist:
                n = 0
                for i in range(self.layout3D):
                    if n < len(pathlist):
                        self.maingrids1.itemAtPosition(i, 0).imagelist.setCurrentIndex(n+1)
                        n+=1
                    else:
                        break

    def link_mode(self):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked and not self.maingrids1.itemAtPosition(i, j).skiplink:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.zoom_link.connect(self.zoom_all)
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.move_link.connect(self.move_all)
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.grey_link.connect(self.grey_all)
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.slice_link.connect(self.slice_all)
                        self.maingrids1.itemAtPosition(i, j).skiplink = True # avoid multi link
                        self.maingrids1.itemAtPosition(i, j).skipdis = False
                    elif not self.maingrids1.itemAtPosition(i, j).islinked and not self.maingrids1.itemAtPosition(i, j).skipdis:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.zoom_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.move_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.grey_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.slice_link.disconnect()
                        self.maingrids1.itemAtPosition(i, j).skipdis = True
                        self.maingrids1.itemAtPosition(i, j).skiplink = False
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked and not self.maingrids1.itemAtPosition(i, 0).skiplink:
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.zoom_link.connect(self.zoom_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.move_link.connect(self.move_all)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas1.grey_link.connect(self.grey_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.zoom_link.connect(self.zoom_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.move_link.connect(self.move_all)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas2.grey_link.connect(self.grey_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.zoom_link.connect(self.zoom_all)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.move_link.connect(self.move_all)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas3.grey_link.connect(self.grey_all)
                    self.maingrids1.itemAtPosition(i, 0).skiplink = True
                    self.maingrids1.itemAtPosition(i, 0).skipdis = False
                elif not self.maingrids1.itemAtPosition(i, 0).islinked and not self.maingrids1.itemAtPosition(i, 0).skipdis:
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.zoom_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.move_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).newcanvas1.grey_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.zoom_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.move_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).newcanvas2.grey_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.zoom_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.move_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).newcanvas3.grey_link.disconnect()
                    self.maingrids1.itemAtPosition(i, 0).skipdis = True
                    self.maingrids1.itemAtPosition(i, 0).skiplink = False

    def zoom_all(self, factor):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.linkedZoom(factor)
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked:
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.linkedZoom(factor)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.linkedZoom(factor)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.linkedZoom(factor)

    def move_all(self, movelist):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).Viewpanel.linkedMove(movelist)
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked:
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel1.linkedMove(movelist)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel2.linkedMove(movelist)
                    self.maingrids1.itemAtPosition(i, 0).Viewpanel3.linkedMove(movelist)

    def grey_all(self, glist):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.linked_grey(glist)
        else:
            for i in range(self.layout3D):
                if self.maingrids1.itemAtPosition(i, 0).islinked:
                    self.maingrids1.itemAtPosition(i, 0).newcanvas1.linked_grey(glist)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas2.linked_grey(glist)
                    self.maingrids1.itemAtPosition(i, 0).newcanvas3.linked_grey(glist)

    def slice_all(self, data):
        if self.gridsnr == 2:
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if self.maingrids1.itemAtPosition(i, j).islinked:
                        self.maingrids1.itemAtPosition(i, j).anewcanvas.linked_slice(data)

    def loadMR(self):
        with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
        self.PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "open file", dbinfo.sPathIn)
        if self.PathDicom:
            self.selectorPath = self.PathDicom
            self.i = self.i + 1
            self.openfile.setDisabled(True)
            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            self.overlay.setGeometry(QtCore.QRect(950, 400, 171, 141))
            self.overlay.show()
            self.newMR = loadImage(self.PathDicom)
            self.newMR.trigger.connect(self.load_end)
            self.newMR.start()
        else:
            pass

    def load_end(self):
        self.overlay.killTimer(self.overlay.timer)
        self.overlay.hide()
        # self.plot_3d(self.newMR.svoxel, 200)
        pathlist.append(self.PathDicom)
        list1.append(self.newMR.voxel_ndarray)
        shapelist.append(self.newMR.new_shape)
        self.mrinmain = self.newMR.voxel_ndarray
        self.NS = self.newMR.new_shape
        self.slices = self.mrinmain.shape[2]
        self.ind = self.slices // 2
        # 40
        self.slices2 = self.mrinmain.shape[0]
        self.ind2 = self.slices2 // 2
        self.slices3 = self.mrinmain.shape[1]
        self.ind3 = self.slices3 // 2
        indlist.clear()
        indlist.append(self.ind)
        ind2list.clear()
        ind2list.append(self.ind2)
        ind3list.clear()
        ind3list.append(self.ind3)
        if self.selectoron == False:
            self.openfile.setDisabled(False)
            if self.gridson == True:
                if self.gridsnr == 2:
                    for i in range(self.layoutlines):
                        for j in range(self.layoutcolumns):
                            if self.selectoron == False:
                                self.maingrids1.itemAtPosition(i, j).addPathd(self.PathDicom,0)
                            else:
                                self.maingrids1.itemAtPosition(i, j).addPathd(self.PathDicom, 1)
                    for i in range(self.layoutlines):
                        for j in range(self.layoutcolumns):
                            if self.maingrids1.itemAtPosition(i, j).mode == 1 and \
                                    self.maingrids1.itemAtPosition(i, j).pathbox.currentIndex() == 0:
                                self.maingrids1.itemAtPosition(i, j).pathbox.setCurrentIndex(len(pathlist))
                                break
                        else:
                            continue
                        break
                else:
                    for i in range(self.layout3D):
                        self.maingrids1.itemAtPosition(i, 0).addPathim(self.PathDicom)
                    for i in range(self.layout3D):
                        if self.maingrids1.itemAtPosition(i, 0).vmode == 1 and \
                                self.maingrids1.itemAtPosition(i, 0).imagelist.currentIndex() == 0:
                            self.maingrids1.itemAtPosition(i, 0).imagelist.setCurrentIndex(len(pathlist))
                            break
            else:
                pass
        else:
            self.load_select()

    def unpatching2(self, result, orig):
        PatchSize = np.array((40.0, 40.0))
        PatchOverlay = 0.5
        imglay = fUnpatch2D(result, PatchSize, PatchOverlay, orig.shape)
        return imglay

    def unpatching8(self, result, orig):
        PatchSize = np.array((40.0, 40.0))
        PatchOverlay = 0.5
        IndexArte = np.argmax(result, 1)
        Arte1, Arte2, Arte3 = UnpatchArte8(IndexArte, PatchSize, PatchOverlay, orig.shape)
        return Arte1, Arte2, Arte3

    def unpatching11(self, result, orig):
        PatchSize = np.array((40.0, 40.0))  ##
        PatchOverlay = 0.5

        IndexType = np.argmax(result, 1)
        IndexType[IndexType == 0] = 1
        IndexType[(IndexType > 1) & (IndexType < 4)] = 2
        IndexType[(IndexType > 6) & (IndexType < 9)] = 3
        IndexType[(IndexType > 3) & (IndexType < 7)] = 4
        IndexType[IndexType > 8] = 5

        from tensorflow.python.data.experimental import Counter
        a = Counter(IndexType).most_common(1)
        domain = a[0][0]

        PType = np.delete(result, [1, 3, 5, 6, 8, 10], 1)  # only 5 region left
        PArte = np.delete(result, [0, 2, 4, 7, 9], 1)
        PArte[:, [4, 5]] = PArte[:, [5, 4]]
        PNew = np.concatenate((PType, PArte), axis=1)
        IndexArte = np.argmax(PNew, 1)

        Type = UnpatchType(IndexType, domain, PatchSize, PatchOverlay, orig.shape)
        Arte = UnpatchArte(IndexArte, PatchSize, PatchOverlay, orig.shape)
        return Type, Arte


    def load_patch(self):
        # resultfile = QtWidgets.QFileDialog.getOpenFileName(self, 'choose the result file', '',
        #         'mat files(*.mat);;h5 files(*.h5);;HDF5 files(*.hdf5)', None, QtWidgets.QFileDialog.DontUseNativeDialog)[0]
        resultfile = QtWidgets.QFileDialog.getOpenFileName(self, 'choose the result file', '', 'All Files (*)', None,QtWidgets.QFileDialog.DontUseNativeDialog)[0]

        if resultfile:
            with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
            PathDicom = QtWidgets.QFileDialog.getExistingDirectory(self, "choose the corresponding image", dbinfo.sPathIn)
            if PathDicom in pathlist:
                n = pathlist.index(PathDicom)
                correslist.append(n)
                conten = sio.loadmat(resultfile)
                if 'prob_pre' in conten:
                    cnum = np.array(conten['prob_pre'])
                    IType, IArte = self.unpatching11(conten['prob_pre'], list1[n])
                    # if IType[0] - list1[n][0] <= PatchSize/2 and IType[1] - list1[n][1] <= PatchSize/2:
                    # else:
                    #     QtWidgets.QMessageBox.information(self, 'Warning', 'Please choose the right file!')
                    #     break
                    problist.append(IType)
                    hatchlist.append(IArte)
                    cnrlist.append(11)
                else:
                    pred = conten['prob_test']
                    pred = pred[0:4320, :]
                    cnum = np.array(pred)
                    if cnum.shape[1] == 2:
                        IType = self.unpatching2(pred, list1[n])

                        problist.append(IType)
                        hatchlist.append(empty1)
                        cnrlist.append(2)
                    # elif cnum.shape[1] == 8: 

                nameofCfile = os.path.split(resultfile)[1]
                nameofCfile = nameofCfile + '   class:' + str(cnum.shape[1])
                pnamelist.append(nameofCfile)
                if self.gridsnr == 3:  #############
                    for i in range(self.layout3D):
                        self.maingrids1.itemAtPosition(i, 0).addPathre(nameofCfile)
                else:
                    for i in range(self.layoutlines):
                        for j in range(self.layoutcolumns):
                            self.maingrids1.itemAtPosition(i, j).addPathre(nameofCfile)

            else:
                QtWidgets.QMessageBox.information(self, 'Warning', 'Please load the original file first!')
        else:
            pass

    def set_color(self):
        c1, c3, h1, h2, v1, v3, ok = Patches_window.getData()
        if ok:
            global cmap1, cmap3, hmap1, hmap2, vtr1, vtr3
            cmap1 = c1
            cmap3 = c3
            hmap1 = h1
            hmap2 = h2
            vtr1 = v1
            vtr3 = v3

    def default_color(self):
        c1, c3, h1, h2, v1, v3, ok = cPre_window.getData()
        if ok:
            global cmap1, cmap3, hmap1, hmap2, vtr1, vtr3
            cmap1 = c1
            cmap3 = c3
            hmap1 = h1
            hmap2 = h2
            vtr1 = v1
            vtr3 = v3

    def newShape(self):
        self.canvasl = self.view_group.anewcanvas
        if self.canvasl.get_open_dialog():

            color = colors.to_hex('b',keep_alpha=True)

            if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
                self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.canvasl.set_open_dialog(False)
                self.prevLabelText = text

                if text is not None:
                    if text not in self.labelHist:
                        self.labelHist.append(text)
                    else:
                        pass

                    self.df = pandas.read_csv(self.labelfile)
                    df_size = pandas.DataFrame.count(self.df)
                    df_rows = df_size['labelshape']-1
                    self.df.loc[df_rows, 'image'] = self.view_group.current_image()
                    self.df.loc[df_rows, 'labelname'] = self.prevLabelText
                    if self.labelDialog.getColor():
                        color = self.labelDialog.getColor()
                        self.df.loc[df_rows, 'labelcolor'] = color
                    else:
                        self.df.loc[df_rows, 'labelcolor'] = color
                    self.df.to_csv(self.labelfile, index=False)

            else:
                text = self.defaultLabelTextLine.text()
                self.prevLabelText = text
                if text not in self.labelHist:
                    self.labelHist.append(text)
                if text is not None:
                    self.df = pandas.read_csv(self.labelfile)
                    df_size = pandas.DataFrame.count(self.df)
                    df_rows = df_size['labelshape']-1
                    self.df.loc[df_rows, 'image'] = self.view_group.current_image()
                    self.df.loc[df_rows, 'labelname'] = self.prevLabelText
                    self.df.loc[df_rows, 'labelcolor'] = color
                    self.df.to_csv(self.labelfile, index=False)

            self.canvasl.set_facecolor(color)
            self.updateList()

    def shapeSelectionChanged(self, selected):
        #selected shape changed --> selected item changes at the same time
        self.canvasl = self.view_group.anewcanvas
        if selected:
            for i in range(0, len(self.labelList.get_list())):
                if self.labelList.get_list()[i][0] == str(self.canvasl.get_selected()):
                    selectedRow = i
                    if not selectedRow is None:
                        self.labelList.selectRow(selectedRow)

    def selector_mode(self):

        if self.selectoron == False:
            self.selectoron = True
            icon2 = QtGui.QIcon()
            icon2.addPixmap(QtGui.QPixmap(":/icons/Icons/switchon.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
            self.bselectoron.setIcon(icon2)
            self.cursorCross.setDisabled(False)
            self.brectangle.setDisabled(False)
            self.bellipse.setDisabled(False)
            self.blasso.setDisabled(False)
            # self.bchoosemark.setDisabled(False)
            self.bnoselect.setDisabled(False)
            self.deleteButton.setDisabled(False)
            self.openlabelButton.setDisabled(False)
            self.viewallButton.setDisabled(False)
            self.infoButton.setDisabled(False)
            self.useDefaultLabelCheckbox.setDisabled(False)
            self.defaultLabelTextLine.setDisabled(False)
            self.labellistBox.setDisabled(False)
            self.add_labelFile()
            self.bdswitch.setDisabled(True)
            self.linebox.setDisabled(True)
            self.columnbox.setDisabled(True)
            self.bgrids.setDisabled(True)
            self.imageShow.setDisabled(True)
            self.openfile.setDisabled(True)
            self.bpatch.setDisabled(True)
            self.bsetcolor.setDisabled(True)
            self.view_group.setlinkoff(True)
            self.actionOpen_file.setEnabled(False)
            self.actionSave.setEnabled(True)
            self.actionLoad.setEnabled(False)
            self.load_mark()
            self.labelList.setDisabled(False)
            self.labelList = LabelTable(self)
            self.labelList.set_table_model(labelfile=self.labelfile, imagefile=self.view_group.current_image())
            self.labelList.setSelectionMode(QAbstractItemView.SingleSelection)
            self.labelList.setSelectionBehavior(QAbstractItemView.SelectRows)
            # bind cell click to a method reference
            self.labelList.clicked.connect(self.labelList.get_selectRow)
            self.labelList.doubleClicked.connect(self.edit_label)
            self.labelList.setSortingEnabled(True)
            self.labelList.resizeRowsToContents()
            self.gridLayout_2.addWidget(self.labelList, 22, 0, 1, 2)
            self.labelList.statusChanged.connect(self.label_on_off)
            self.labelList.selectChanged.connect(self.labelItemChanged)

        else:
            self.selectoron = False
            icon1 = QtGui.QIcon()
            icon1.addPixmap(QtGui.QPixmap(":/icons/Icons/switchoff.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.bselectoron.setIcon(icon1)
            self.cursorCross.setDisabled(True)
            self.brectangle.setDisabled(True)
            self.bellipse.setDisabled(True)
            self.blasso.setDisabled(True)
            # self.bchoosemark.setDisabled(True)
            self.bnoselect.setDisabled(True)
            self.deleteButton.setDisabled(True)
            self.openlabelButton.setDisabled(True)
            self.viewallButton.setDisabled(True)
            self.infoButton.setDisabled(True)
            self.useDefaultLabelCheckbox.setDisabled(True)
            self.defaultLabelTextLine.setDisabled(True)
            self.labellistBox.setDisabled(True)
            self.bdswitch.setDisabled(False)
            self.linebox.setDisabled(False)
            self.bgrids.setDisabled(False)
            self.imageShow.setDisabled(False)
            self.openfile.setDisabled(False)
            self.bpatch.setDisabled(False)
            self.bsetcolor.setDisabled(False)
            self.view_group.setlinkoff(False)
            self.actionOpen_file.setEnabled(True)
            self.actionSave.setEnabled(True)
            self.actionLoad.setEnabled(True)
            self.labelList.setDisabled(True)
            self.clear_markings()

            # if self.vision == 2:
            #     self.columnbox.setDisabled(False)
            # for i in reversed(range(self.maingrids1.count())):
            #     self.maingrids1.itemAt(i).widget().setParent(None)

    def stop_view(self, n):
        self.view_group.Viewpanel.stopMove(n)

    def load_mark(self):
        if self.PathDicom:

            self.overlay = Overlay(self.centralWidget())
            self.overlay.setGeometry(QtCore.QRect(950, 400, 171, 141))
            self.overlay.show()
            self.newMR = loadImage(self.selectorPath)
            self.newMR.trigger.connect(self.load_end)
            self.newMR.start()


    def edit_label(self, ind):

        self.canvasl = self.view_group.anewcanvas

        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)
        text = self.labelDialog.popUp(text=self.prevLabelText)
        self.canvasl.set_open_dialog(False)
        self.prevLabelText = text

        if text is not None:
            if text not in self.labelHist:
                self.labelHist.append(text)
            else:
                pass

            self.selectind = ind.row()
            self.df = pandas.read_csv(self.labelfile)
            self.df.loc[self.selectind, 'labelname'] = self.prevLabelText
            if self.labelDialog.getColor():
                color = self.labelDialog.getColor()
                self.df.loc[self.selectind, 'labelcolor'] = color
            else:
                color = self.df.loc[self.selectind, 'labelcolor']
            self.df.to_csv(self.labelfile, index=False)
            self.canvasl.set_facecolor(color)
            self.updateList()

    def show_label_info(self):
        df = pandas.read_csv(self.labelfile)
        gtabview.view(df)

    def save_label(self):
        list = np.array(self.labelList.get_list())
        column = ['artist','labelshape','slice','path','status','image','labelname','labelcolor']
        df = pandas.DataFrame(data=list, index=None, columns=column)
        try:
            save_dialog = QtWidgets.QFileDialog()
            save_dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            file_path = save_dialog.getSaveFileName(self, 'Save as... File', './',
                                                    filter='All Files(*.*);; csv Files(*.csv)')

            if file_path[0]:
                self.file_path = file_path
                file_open = open(self.file_path[0], 'w')
                self.file_name = (self.file_path[0].split('/'))[-1] + '.csv'
                self.setWindowTitle("{} - Please save every label records in a individual folder".format(self.file_name))
                df.to_csv(self.file_path[0] + '.csv')

        except FileNotFoundError as why:
            self.error_box(why)
            pass

    def delete_labelItem(self):
        self.selectind = self.labelList.get_table_model().get_selectind()
        self.df = pandas.read_csv(self.labelfile)
        self.df = self.df.drop(self.df.index[self.selectind])
        self.df.to_csv('Markings/marking_records.csv', index=False)
        self.updateList()
        self.delete_labelShape()

    def delete_labelShape(self):

        self.clear_markings()
        self.load_markings(self.view_group.current_slice())

    def open_labelFile(self):

        path = 'Markings'
        dialog = QtWidgets.QFileDialog()
        path = dialog.getExistingDirectory(self, "open file", path)

        self.labelfile = path + '/' + os.listdir(path)[0]
        if self.labelfile:
            self.add_labelFile()
            self.view_group.modechange.setDisabled(False)
            self.labellistBox.currentTextChanged.connect(self.update_labelFile)
        else:
            pass


    def add_labelFile(self):

        imageItems = [self.labellistBox.itemText(i) for i in range(self.labellistBox.count())]
        region = os.path.split(self.labelfile)
        proband = os.path.split(os.path.split(region[0])[0])[1]
        region = region[1]
        newItem = '[' + (proband) + '][' + (region) + ']'
        if newItem not in imageItems:
            self.labellistBox.addItem(newItem)

    def update_labelFile(self):
        if not self.labellistBox.currentText()=="Label List":
            self.updateList()
            self.clear_markings()
            self.load_markings(self.view_group.current_slice())

    def load_select(self):  # from load_end
        self.vision = 2
        self.clearall()
        self.gridsnr = 2
        self.layoutlines = 1
        self.layoutcolumns = 1
        for i in range(self.layoutlines):
            for j in range(self.layoutcolumns):
                blocklayout = Viewgroup()
                self.view_group = blocklayout
                self.view_group.setlinkoff(True)
                for dpath in pathlist:
                    blocklayout.addPathd(dpath, 1)
                for cpath in pnamelist:
                    blocklayout.addPathre(cpath)
                self.maingrids1.addLayout(blocklayout, i, j)
        if pathlist:
            n = 0
            for i in range(self.layoutlines):
                for j in range(self.layoutcolumns):
                    if n < len(pathlist):
                        self.maingrids1.itemAtPosition(i, j).pathbox.setCurrentIndex(n + 1)
                        n += 1
                    else:
                        break

        self.graylist = []
        self.graylist.append(None)
        self.graylist.append(None)
        self.emitlist = []
        self.emitlist.append(self.ind)
        self.emitlist.append(self.slices)

        self.graylist2 = []
        self.graylist2.append(None)
        self.graylist2.append(None)
        self.emitlist2 = []
        self.emitlist2.append(self.ind2)
        self.emitlist2.append(self.slices2)

        self.graylist3 = []
        self.graylist3.append(None)
        self.graylist3.append(None)
        self.emitlist3 = []
        self.emitlist3.append(self.ind3)
        self.emitlist3.append(self.slices3)

        self.updateList()
        self.load_markings(self.view_group.current_slice())
        self.view_group.rotateSignal.connect(self.load_markings)
        self.view_group.scrollSignal.connect(self.load_markings)
        self.view_group.rotateSignal.connect(self.changeSelector)
        self.view_group.scrollSignal.connect(self.changeSelector)
        self.view_group.zoomSignal.connect(self.changeSelector)

    def changeSelector(self, val):
        self.bnoselect.setChecked(True)
        self.brectangle.setChecked(False)
        self.bellipse.setChecked(False)
        self.blasso.setChecked(False)

    def load_markings(self, slice):

        df = pandas.read_csv(self.labelfile)
        num1 = df[df['image'] == self.view_group.current_image()].index.values.astype(int)
        num2 = df[df['slice'] == slice].index.values.astype(int)
        num = list(set(num1).intersection(num2))
        self.markings = []
        self.canvasl = self.view_group.anewcanvas
        for i in range(0, len(num)):
            status = df['status'][num[i]]
            df.select_dtypes(include='object')
            if df['labelshape'][num[i]]=='lasso':
                path = Path(np.asarray(eval(df['path'][num[i]])))
                newItem = PathPatch(path, fill=True, alpha=.2, edgecolor=None)
            else:
                newItem = eval(df['artist'][num[i]])
            color = df['labelcolor'][num[i]]
            self.markings.append(newItem)
            if status == 0:
                newItem.set_visible(True)
                if type(newItem) is Rectangle or Ellipse:
                    newItem.set_picker(True)
                else:
                    newItem.set_picker(False)
                newItem.set_facecolor(color)
                newItem.set_alpha(0.5)
            else:
                newItem.set_visible(False)
            self.canvasl.ax1.add_artist(newItem)

        self.canvasl.blit(self.canvasl.ax1.bbox)

    def label_on_off(self, status):

        for item in self.markings:
            item.remove()
        self.newcanvasview()
        self.canvasl = self.view_group.anewcanvas
        self.canvasl.draw_idle()
        df = pandas.read_csv(self.labelfile)
        num1 = df[df['image'] == self.view_group.current_image()].index.values.astype(int)
        num2 = df[df['slice'] == self.view_group.current_slice()].index.values.astype(int)
        num = list(set(num1).intersection(num2))
        self.markings = []

        for i in range(0, len(num)):
            status = df['status'][num[i]]
            df.select_dtypes(include='object')
            if df['labelshape'][num[i]] == 'lasso':
                path = Path(np.asarray(eval(df['path'][num[i]])))
                newItem = PathPatch(path, fill=True, alpha=.2, edgecolor=None)
            else:
                newItem = eval(df['artist'][num[i]])
            color = df['labelcolor'][num[i]]
            self.markings.append(newItem)

            if not status:
                newItem.set_visible(True)
                if type(newItem) is Rectangle or Ellipse:
                    newItem.set_picker(True)
                else:
                    newItem.set_picker(False)
                newItem.set_facecolor(color)
                newItem.set_alpha(0.5)
            else:
                newItem.set_visible(False)

            self.canvasl.ax1.add_artist(newItem)

        self.canvasl.blit(self.canvasl.ax1.bbox)

    def clear_markings(self):

        self.load_old()


    def marking_shape(self, n):
        self.canvasl = self.view_group.anewcanvas
        state = self.cursorCross.isChecked()

        if self.selectoron == True:

            if n == 0:
                self.canvasl.set_state(2)
                self.canvasl.rec_toggle_selector_off()
                self.canvasl.ell_toggle_selector_off()
                self.canvasl.lasso_toggle_selector_off()

            elif n == 1:
                self.canvasl.set_state(1)
                self.canvasl.rec_toggle_selector_on()
                self.canvasl.set_cursor(state)
                self.canvasl.ell_toggle_selector_off()
                self.canvasl.lasso_toggle_selector_off()

            elif n == 2:
                self.canvasl.set_state(1)
                self.canvasl.rec_toggle_selector_off()
                self.canvasl.ell_toggle_selector_on()
                self.canvasl.set_cursor(state)
                self.canvasl.lasso_toggle_selector_off()

            elif n == 3:
                self.canvasl.set_state(1)
                self.canvasl.rec_toggle_selector_off()
                self.canvasl.ell_toggle_selector_off()
                self.canvasl.lasso_toggle_selector_on()
                self.canvasl.set_cursor(state)

        else:
            pass

        self.canvasl.newShape.connect(self.newShape)
        self.canvasl.deleteEvent.connect(self.updateList)
        self.canvasl.selectionChanged.connect(self.shapeSelectionChanged)

    def handle_crossline(self):
        self.canvasl = self.view_group.anewcanvas
        state = self.cursorCross.isChecked()
        self.canvasl.set_cursor(state)

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def labelItemChanged(self, item):
        #selected label item changed, selected shape changes at the same time
        self.canvasl = self.view_group.anewcanvas
        shapeitem = self.labelList.get_list()[item]
        if not shapeitem[1] == 'lasso':
            shape = eval(shapeitem[0])
            self.canvasl.set_selected(shape)

    def updateList(self):
        self.labelList.set_table_model(self.labelfile, self.view_group.current_image())

    def newonscroll(self, event):

        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        if self.ind >= self.slices:
            self.ind = 0
        if self.ind <= -1:
            self.ind = self.slices - 1

        self.emitlist[0] = self.ind
        self.update_data.emit(self.emitlist)
        self.newcanvasview()

    def newonscrol2(self, event):

        if event.button == 'up':
            self.ind2 = (self.ind2 + 1) % self.slices2
        else:
            self.ind2 = (self.ind2 - 1) % self.slices2
        if self.ind2 >= self.slices2:
            self.ind2 = 0
        if self.ind2 <= -1:
            self.ind2 = self.slices2 - 1

        self.emitlist2[0] = self.ind2
        self.update_data2.emit(self.emitlist2)
        self.newcanvasview()

    def newonscrol3(self, event):

        if event.button == 'up':
            self.ind3 = (self.ind3 + 1) % self.slices3
        else:
            self.ind3 = (self.ind3 - 1) % self.slices3
        if self.ind3 >= self.slices3:
            self.ind3 = 0
        if self.ind3 <= -1:
            self.ind3 = self.slices3 - 1
        #
        self.emitlist3[0] = self.ind3
        self.update_data3.emit(self.emitlist3)
        self.newcanvasview()

    def newcanvasview(self):  # refreshing

        self.newax.clear()
        self.newax2.clear()
        self.newax3.clear()

        self.pltc = self.newax.imshow(np.swapaxes(self.mrinmain[:, :, self.ind], 0, 1), cmap='gray', vmin=0, vmax=2094)
        self.pltc2 = self.newax2.imshow(np.swapaxes(self.mrinmain[self.ind2, :, :], 0, 1), cmap='gray', vmin=0,
                                        vmax=2094,
                                        extent=[0, self.NS[1], self.NS[2], 0], interpolation='sinc')
        self.pltc3 = self.newax3.imshow(np.swapaxes(self.mrinmain[:, self.ind3, :], 0, 1), cmap='gray', vmin=0,
                                        vmax=2094,
                                        extent=[0, self.NS[0], self.NS[2], 0], interpolation='sinc')

        sepkey = os.path.split(
            self.selectorPath)  # ('C:/Users/hansw/Videos/artefacts/MRPhysics/newProtocol/01_ab/dicom_sorted', 't1_tse_tra_Kopf_Motion_0003')
        sepkey = sepkey[1]  # t1_tse_tra_Kopf_Motion_0003


        self.newcanvas.draw()  # not self.newcanvas.show()
        self.newcanvas2.draw()
        self.newcanvas3.draw()

        v_min, v_max = self.pltc.get_clim()
        self.graylist[0] = v_min
        self.graylist[1] = v_max
        self.new_page.emit()

        v_min, v_max = self.pltc2.get_clim()
        self.graylist2[0] = v_min
        self.graylist2[1] = v_max
        self.new_page2.emit()

        v_min, v_max = self.pltc3.get_clim()
        self.graylist3[0] = v_min
        self.graylist3[1] = v_max
        self.new_page3.emit()

    def deleteLabel(self):
        pass

    def mouse_clicked(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.mouse_second_clicked = True

    def mouse_clicked2(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.mouse_second_clicked = True

    def mouse_clicked3(self, event):
        if event.button == 2:
            self.x_clicked = event.x
            self.y_clicked = event.y
            self.mouse_second_clicked = True
            # todo

    def mouse_move(self, event):

        if self.mouse_second_clicked:
            factor = 10
            __x = event.x - self.x_clicked
            __y = event.y - self.y_clicked
            v_min, v_max = self.pltc.get_clim()
            if __x >= 0 and __y >= 0:
                __vmin = np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y >= 0:
                __vmin = -np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y < 0:
                __vmin = -np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor - np.abs(__y) * factor
            else:
                __vmin = np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = np.abs(__x) * factor - np.abs(__y) * factor

            if (float(__vmin - __vmax)) / (v_max - v_min + 0.001) > 1:
                nmb = (float(__vmin - __vmax)) / (v_max - v_min + 0.001) + 1
                __vmin = (float(__vmin - __vmax)) / nmb * (__vmin / (__vmin - __vmax))
                __vmax = (float(__vmin - __vmax)) / nmb * (__vmax / (__vmin - __vmax))

            v_min += __vmin
            v_max += __vmax
            if v_min < v_max:
                self.pltc.set_clim(vmin=v_min, vmax=v_max)
                self.graylist[0] = v_min.round(2)
                self.graylist[1] = v_max.round(2)
                self.gray_data.emit(self.graylist)

                self.newcanvas.draw_idle()
            else:
                v_min -= __vmin
                v_max -= __vmax

    def mouse_move2(self, event):

        if self.mouse_second_clicked:
            factor = 10
            __x = event.x - self.x_clicked
            __y = event.y - self.y_clicked
            v_min, v_max = self.pltc2.get_clim()
            if __x >= 0 and __y >= 0:
                __vmin = np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y >= 0:
                __vmin = -np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y < 0:
                __vmin = -np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor - np.abs(__y) * factor
            else:
                __vmin = np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = np.abs(__x) * factor - np.abs(__y) * factor

            if (float(__vmin - __vmax)) / (v_max - v_min + 0.001) > 1:
                nmb = (float(__vmin - __vmax)) / (v_max - v_min + 0.001) + 1
                __vmin = (float(__vmin - __vmax)) / nmb * (__vmin / (__vmin - __vmax))
                __vmax = (float(__vmin - __vmax)) / nmb * (__vmax / (__vmin - __vmax))

            v_min += __vmin
            v_max += __vmax
            if v_min < v_max:
                self.pltc2.set_clim(vmin=v_min, vmax=v_max)
                self.graylist2[0] = v_min.round(2)
                self.graylist2[1] = v_max.round(2)
                self.gray_data2.emit(self.graylist2)

                self.newcanvas2.draw_idle()
            else:
                v_min -= __vmin
                v_max -= __vmax

    def mouse_move3(self, event):

        if self.mouse_second_clicked:
            factor = 10
            __x = event.x - self.x_clicked
            __y = event.y - self.y_clicked
            v_min, v_max = self.pltc3.get_clim()
            if __x >= 0 and __y >= 0:
                __vmin = np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y >= 0:
                __vmin = -np.abs(__x) * factor - np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor + np.abs(__y) * factor
            elif __x < 0 and __y < 0:
                __vmin = -np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = -np.abs(__x) * factor - np.abs(__y) * factor
            else:
                __vmin = np.abs(__x) * factor + np.abs(__y) * factor
                __vmax = np.abs(__x) * factor - np.abs(__y) * factor

            if (float(__vmin - __vmax)) / (v_max - v_min + 0.001) > 1:
                nmb = (float(__vmin - __vmax)) / (v_max - v_min + 0.001) + 1
                __vmin = (float(__vmin - __vmax)) / nmb * (__vmin / (__vmin - __vmax))
                __vmax = (float(__vmin - __vmax)) / nmb * (__vmax / (__vmin - __vmax))

            v_min += __vmin
            v_max += __vmax
            if v_min < v_max:
                self.pltc3.set_clim(vmin=v_min, vmax=v_max)
                self.graylist3[0] = v_min.round(2)
                self.graylist3[1] = v_max.round(2)
                self.gray_data3.emit(self.graylist3)

                self.newcanvas3.draw_idle()
            else:
                v_min -= __vmin
                v_max -= __vmax

    def mouse_release(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

    def mouse_release2(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

    def mouse_release3(self, event):
        if event.button == 2:
            self.mouse_second_clicked = False

    def newSliceview(self):
        self.graylabel.setText('G %s' % (self.graylist))

    def updateSlices(self, elist):
        self.slicelabel.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))
        indlist.append(elist[0])

    def updateGray(self, elist):
        self.graylabel.setText('G %s' % (elist))

    def updateZoom(self, factor):
        self.zoomlabel.setText('XY %s' % (factor))

    def setGreymain(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            self.pltc.set_clim(vmin=minv, vmax=maxv)
            self.pltc2.set_clim(vmin=minv, vmax=maxv)
            self.pltc3.set_clim(vmin=minv, vmax=maxv)
            self.graylist[0] = minv
            self.graylist[1] = maxv
            self.gray_data.emit(self.graylist)
            self.newcanvas.draw_idle()
            self.gray_data2.emit(self.graylist)
            self.newcanvas2.draw_idle()
            self.gray_data3.emit(self.graylist)
            self.newcanvas3.draw_idle()

    def newSliceview2(self):

        self.graylabel2.setText('G %s' % (self.graylist2))

    def updateSlices2(self, elist):
        self.slicelabel2.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateGray2(self, elist):
        self.graylabel2.setText('G %s' % (elist))

    def updateZoom2(self, factor):
        self.zoomlabel2.setText('YZ %s' % (factor))

    def setGreymain2(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            self.pltc.set_clim(vmin=minv, vmax=maxv)
            self.pltc2.set_clim(vmin=minv, vmax=maxv)
            self.pltc3.set_clim(vmin=minv, vmax=maxv)
            self.graylist2[0] = minv
            self.graylist2[1] = maxv
            self.gray_data.emit(self.graylist)
            self.newcanvas.draw_idle()
            self.gray_data2.emit(self.graylist)
            self.newcanvas2.draw_idle()
            self.gray_data3.emit(self.graylist)
            self.newcanvas3.draw_idle()

    def newSliceview3(self):

        self.graylabel3.setText('G %s' % (self.graylist3))

    def updateSlices3(self, elist):
        self.slicelabel3.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))

    def updateGray3(self, elist):
        self.graylabel3.setText('G %s' % (elist))

    def updateZoom3(self, factor):
        self.zoomlabel3.setText('XZ %s' % (factor))

    def setGreymain3(self):
        maxv, minv, ok = grey_window.getData()
        if ok:
            self.pltc.set_clim(vmin=minv, vmax=maxv)
            self.pltc2.set_clim(vmin=minv, vmax=maxv)
            self.pltc3.set_clim(vmin=minv, vmax=maxv)
            self.graylist3[0] = minv
            self.graylist3[1] = maxv
            self.gray_data.emit(self.graylist)
            self.newcanvas.draw_idle()
            self.gray_data2.emit(self.graylist)
            self.newcanvas2.draw_idle()
            self.gray_data3.emit(self.graylist)
            self.newcanvas3.draw_idle()


    def closeEvent(self, QCloseEvent):
        reply = QtWidgets.QMessageBox.question(self, 'Warning', 'Are you sure to exit?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


####### second tab
    def button_train_clicked(self):
        # set epochs
        self.deepLearningArtApp.setEpochs(self.SpinBox_Epochs.value())

        # handle check states of check boxes for used classes
        self.deepLearningArtApp.setUsingArtifacts(self.CheckBox_Artifacts.isChecked())
        self.deepLearningArtApp.setUsingBodyRegions(self.CheckBox_BodyRegion.isChecked())
        self.deepLearningArtApp.setUsingTWeighting(self.CheckBox_TWeighting.isChecked())

        # set learning rates and batch sizes
        try:
            batchSizes = np.fromstring(self.LineEdit_BatchSizes.text(), dtype=np.int, sep=',')
            self.deepLearningArtApp.setBatchSizes(batchSizes)
            learningRates = np.fromstring(self.LineEdit_LearningRates.text(), dtype=np.float32, sep=',')
            self.deepLearningArtApp.setLearningRates(learningRates)
        except:
            raise ValueError("Wrong input format of learning rates! Enter values seperated by ','. For example: 0.1,0.01,0.001")

        # set optimizer
        selectedOptimizer = self.ComboBox_Optimizers.currentText()
        if selectedOptimizer == "SGD":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.SGD_OPTIMIZER)
        elif selectedOptimizer == "RMSprop":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.RMS_PROP_OPTIMIZER)
        elif selectedOptimizer == "Adagrad":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.ADAGRAD_OPTIMIZER)
        elif selectedOptimizer == "Adadelta":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.ADADELTA_OPTIMIZER)
        elif selectedOptimizer == "Adam":
            self.deepLearningArtApp.setOptimizer(DeepLearningArtApp.ADAM_OPTIMIZER)
        else:
            raise ValueError("Unknown Optimizer!")

        # set weigth decay
        self.deepLearningArtApp.setWeightDecay(float(self.DoubleSpinBox_WeightDecay.value()))
        # set momentum
        self.deepLearningArtApp.setMomentum(float(self.DoubleSpinBox_Momentum.value()))
        # set nesterov enabled
        if self.CheckBox_Nesterov.checkState() == QtCore.Qt.Checked:
            self.deepLearningArtApp.setNesterovEnabled(True)
        else:
            self.deepLearningArtApp.setNesterovEnabled(False)

        # handle data augmentation
        if self.CheckBox_DataAugmentation.checkState() == QtCore.Qt.Checked:
            self.deepLearningArtApp.setDataAugmentationEnabled(True)
            # get all checked data augmentation options
            if self.CheckBox_DataAug_horizontalFlip.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setHorizontalFlip(True)
            else:
                self.deepLearningArtApp.setHorizontalFlip(False)

            if self.CheckBox_DataAug_verticalFlip.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setVerticalFlip(True)
            else:
                self.deepLearningArtApp.setVerticalFlip(False)

            if self.CheckBox_DataAug_Rotation.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setRotation(True)
            else:
                self.deepLearningArtApp.setRotation(False)

            if self.CheckBox_DataAug_zcaWeighting.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setZCA_Whitening(True)
            else:
                self.deepLearningArtApp.setZCA_Whitening(False)

            if self.CheckBox_DataAug_HeightShift.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setHeightShift(True)
            else:
                self.deepLearningArtApp.setHeightShift(False)

            if self.CheckBox_DataAug_WidthShift.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setWidthShift(True)
            else:
                self.deepLearningArtApp.setWidthShift(False)

            if self.CheckBox_DataAug_Zoom.checkState() == QtCore.Qt.Checked:
                self.deepLearningArtApp.setZoom(True)
            else:
                self.deepLearningArtApp.setZoom(False)


            # contrast improvement (contrast stretching, adaptive equalization, histogram equalization)
            # it is not recommended to set more than one of them to true
            if self.RadioButton_DataAug_contrastStretching.isChecked():
                self.deepLearningArtApp.setContrastStretching(True)
            else:
                self.deepLearningArtApp.setContrastStretching(False)

            if self.RadioButton_DataAug_histogramEq.isChecked():
                self.deepLearningArtApp.setHistogramEqualization(True)
            else:
                self.deepLearningArtApp.setHistogramEqualization(False)

            if self.RadioButton_DataAug_adaptiveEq.isChecked():
                self.deepLearningArtApp.setAdaptiveEqualization(True)
            else:
                self.deepLearningArtApp.setAdaptiveEqualization(False)
        else:
            # disable data augmentation
            self.deepLearningArtApp.setDataAugmentationEnabled(False)


        # start training process
        self.deepLearningArtApp.performTraining()



    def button_markingsPath_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getMarkingsPath())
        self.Label_MarkingsPath.setText(dir)
        self.deepLearningArtApp.setMarkingsPath(dir)



    def button_patching_clicked(self):
        if self.deepLearningArtApp.getSplittingMode() == DeepLearningArtApp.NONE_SPLITTING:
            QtWidgets.QMessageBox.about(self, "My message box", "Select Splitting Mode!")
            return 0

        self.getSelectedDatasets()
        self.getSelectedPatients()

        # get patching parameters
        self.deepLearningArtApp.setPatchSizeX(self.SpinBox_PatchX.value())
        self.deepLearningArtApp.setPatchSizeY(self.SpinBox_PatchY.value())
        self.deepLearningArtApp.setPatchSizeZ(self.SpinBox_PatchZ.value())
        self.deepLearningArtApp.setPatchOverlapp(self.SpinBox_PatchOverlapp.value())

        # get labling parameters
        if self.RadioButton_MaskLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(DeepLearningArtApp.MASK_LABELING)
        elif self.RadioButton_PatchLabeling.isChecked():
            self.deepLearningArtApp.setLabelingMode(DeepLearningArtApp.PATCH_LABELING)

        # get patching parameters
        if self.ComboBox_Patching.currentIndex() == 1:
            # 2D patching selected
            self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_2D)
        elif self.ComboBox_Patching.currentIndex() == 2:
            # 3D patching selected
            self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_3D)
        else:
            self.ComboBox_Patching.setCurrentIndex(1)
            self.deepLearningArtApp.setPatchingMode(DeepLearningArtApp.PATCHING_2D)

        #using segmentation mask
        self.deepLearningArtApp.setUsingSegmentationMasks(self.CheckBox_SegmentationMask.isChecked())

        # handle store mode
        self.deepLearningArtApp.setStoreMode(self.ComboBox_StoreOptions.currentIndex())

        print("Start Patching for ")
        print("the Patients:")
        for x in self.deepLearningArtApp.getSelectedPatients():
            print(x)
        print("and the Datasets:")
        for x in self.deepLearningArtApp.getSelectedDatasets():
            print(x)
        print("with the following Patch Parameters:")
        print("Patch Size X: " + str(self.deepLearningArtApp.getPatchSizeX()))
        print("Patch Size Y: " + str(self.deepLearningArtApp.getPatchSizeY()))
        print("Patch Overlapp: " + str(self.deepLearningArtApp.getPatchOverlapp()))

        #generate dataset
        self.deepLearningArtApp.generateDataset()

        #check if attributes in DeepLearningArtApp class contains dataset
        if self.deepLearningArtApp.datasetAvailable() == True:
            # if yes, make the use current data button available
            self.Button_useCurrentData.setEnabled(True)


    def button_outputPatching_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        self.Label_OutputPathPatching.setText(dir)
        self.deepLearningArtApp.setOutputPathForPatching(dir)

    def getSelectedPatients(self):
        selectedPatients = []
        for i in range(self.TreeWidget_Patients.topLevelItemCount()):
            if self.TreeWidget_Patients.topLevelItem(i).checkState(0) == QtCore.Qt.Checked:
                selectedPatients.append(self.TreeWidget_Patients.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedPatients(selectedPatients)

    def button_DB_clicked(self):
        dir = self.openFileNamesDialog(self.deepLearningArtApp.getPathToDatabase())
        self.deepLearningArtApp.setPathToDatabase(dir)
        self.manageTreeView()

    def openFileNamesDialog(self, dir=None):
        if dir==None:
            with open('config' + os.sep + 'param.yml', 'r') as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            dbinfo = DatabaseInfo(cfg['MRdatabase'], cfg['subdirs'])
            dir = dbinfo.sPathIn + os.sep + 'MRPhysics'  + os.sep + 'newProtocol'

        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog

        ret = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", dir)
        # path to database
        dir = str(ret)
        return dir

    def manageTreeView(self):
        # all patients in database
        if os.path.exists(self.deepLearningArtApp.getPathToDatabase()):
            subdirs = os.listdir(self.deepLearningArtApp.getPathToDatabase())
            self.TreeWidget_Patients.setHeaderLabel("Patients:")

            for x in subdirs:
                item = QtWidgets.QTreeWidgetItem()
                item.setText(0, str(x))
                item.setCheckState(0, QtCore.Qt.Unchecked)
                self.TreeWidget_Patients.addTopLevelItem(item)

            self.Label_DB.setText(self.deepLearningArtApp.getPathToDatabase())

    def manageTreeViewDatasets(self):
        # print(os.path.dirname(self.deepLearningArtApp.getPathToDatabase()))
        # manage datasets
        self.TreeWidget_Datasets.setHeaderLabel("Datasets:")
        for ds in DeepLearningArtApp.datasets.keys():
            dataset = DeepLearningArtApp.datasets[ds].getPathdata()
            item = QtWidgets.QTreeWidgetItem()
            item.setText(0, dataset)
            item.setCheckState(0, QtCore.Qt.Unchecked)
            self.TreeWidget_Datasets.addTopLevelItem(item)

    def getSelectedDatasets(self):
        selectedDatasets = []
        for i in range(self.TreeWidget_Datasets.topLevelItemCount()):
            if self.TreeWidget_Datasets.topLevelItem(i).checkState(0) == QtCore.Qt.Checked:
                selectedDatasets.append(self.TreeWidget_Datasets.topLevelItem(i).text(0))

        self.deepLearningArtApp.setSelectedDatasets(selectedDatasets)

    def selectedDNN_changed(self):
        self.deepLearningArtApp.setNeuralNetworkModel(self.ComboBox_DNNs.currentText())

    def button_useCurrentData_clicked(self):
        if self.deepLearningArtApp.datasetAvailable() == True:
            self.Label_currentDataset.setText("Current Dataset is used...")
            self.GroupBox_TrainNN.setEnabled(True)
        else:
            self.Button_useCurrentData.setEnabled(False)
            self.Label_currentDataset.setText("No Dataset selected!")
            self.GroupBox_TrainNN.setEnabled(False)

    def button_selectDataset_clicked(self):
        pathToDataset = self.openFileNamesDialog(self.deepLearningArtApp.getOutputPathForPatching())
        retbool, datasetName = self.deepLearningArtApp.loadDataset(pathToDataset)
        if retbool == True:
            self.Label_currentDataset.setText(datasetName + " is used as dataset...")
        else:
            self.Label_currentDataset.setText("No Dataset selected!")

        if self.deepLearningArtApp.datasetAvailable() == True:
            self.GroupBox_TrainNN.setEnabled(True)
        else:
            self.GroupBox_TrainNN.setEnabled(False)

    def button_learningOutputPath_clicked(self):
        path = self.openFileNamesDialog(self.deepLearningArtApp.getLearningOutputPath())
        self.deepLearningArtApp.setLearningOutputPath(path)
        self.Label_LearningOutputPath.setText(path)

    def updateProgressBarTraining(self, val):
        self.ProgressBar_training.setValue(val)

    def splittingMode_changed(self):

        if self.ComboBox_splittingMode.currentIndex() == 0:
            self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
            self.Label_SplittingParams.setText("Select splitting mode!")
        elif self.ComboBox_splittingMode.currentIndex() == 1:
            # call input dialog for editting ratios
            testTrainingRatio, retBool = QtWidgets.QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
                                                             "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)
            if retBool == True:
                validationTrainingRatio, retBool = QtWidgets.QInputDialog.getDouble(self, "Enter Validation/Training Ratio",
                                                                      "Ratio Validation/Training Set: ", 0.2, 0, 1, decimals=2)
                if retBool == True:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.SIMPLE_RANDOM_SAMPLE_SPLITTING)
                    self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
                    self.deepLearningArtApp.setTrainValidationRatio(validationTrainingRatio)
                    txtStr = "using Test/Train=" + str(testTrainingRatio) + " and Valid/Train=" + str(validationTrainingRatio)
                    self.Label_SplittingParams.setText(txtStr)
                else:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                    self.ComboBox_splittingMode.setCurrentIndex(0)
                    self.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                self.ComboBox_splittingMode.setCurrentIndex(0)
                self.Label_SplittingParams.setText("Select Splitting Mode!")
        elif self.ComboBox_splittingMode.currentIndex() == 2:
            # cross validation splitting
            testTrainingRatio, retBool = QtWidgets.QInputDialog.getDouble(self, "Enter Test/Training Ratio:",
                                                             "Ratio Test/Training Set:", 0.2, 0, 1, decimals=2)

            if retBool == True:
                numFolds, retBool = QtWidgets.QInputDialog.getInt(self, "Enter Number of Folds for Cross Validation",
                                                    "Number of Folds: ", 15, 0, 100000)
                if retBool == True:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.CROSS_VALIDATION_SPLITTING)
                    self.deepLearningArtApp.setTrainTestDatasetRatio(testTrainingRatio)
                    self.deepLearningArtApp.setNumFolds(numFolds)
                    self.Label_SplittingParams.setText("Test/Train Ratio: " + str(testTrainingRatio) + \
                                                          ", and " + str(numFolds) + " Folds")
                else:
                    self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                    self.ComboBox_splittingMode.setCurrentIndex(0)
                    self.Label_SplittingParams.setText("Select Splitting Mode!")
            else:
                self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.NONE_SPLITTING)
                self.ComboBox_splittingMode.setCurrentIndex(0)
                self.Label_SplittingParams.setText("Select Splitting Mode!")

        elif self.ComboBox_splittingMode.currentIndex() == 3:
            self.deepLearningArtApp.setSplittingMode(DeepLearningArtApp.PATIENT_CROSS_VALIDATION_SPLITTING)

    def check_dataAugmentation_enabled(self):
        if self.CheckBox_DataAugmentation.checkState() == QtCore.Qt.Checked:
            self.CheckBox_DataAug_horizontalFlip.setEnabled(True)
            self.CheckBox_DataAug_verticalFlip.setEnabled(True)
            self.CheckBox_DataAug_Rotation.setEnabled(True)
            self.CheckBox_DataAug_zcaWeighting.setEnabled(True)
            self.CheckBox_DataAug_HeightShift.setEnabled(True)
            self.CheckBox_DataAug_WidthShift.setEnabled(True)
            self.CheckBox_DataAug_Zoom.setEnabled(True)
            self.RadioButton_DataAug_contrastStretching.setEnabled(True)
            self.RadioButton_DataAug_histogramEq.setEnabled(True)
            self.RadioButton_DataAug_adaptiveEq.setEnabled(True)
        else:
            self.CheckBox_DataAug_horizontalFlip.setEnabled(False)
            self.CheckBox_DataAug_verticalFlip.setEnabled(False)
            self.CheckBox_DataAug_Rotation.setEnabled(False)
            self.CheckBox_DataAug_zcaWeighting.setEnabled(False)
            self.CheckBox_DataAug_HeightShift.setEnabled(False)
            self.CheckBox_DataAug_WidthShift.setEnabled(False)
            self.CheckBox_DataAug_Zoom.setEnabled(False)

            self.RadioButton_DataAug_contrastStretching.setEnabled(False)
            self.RadioButton_DataAug_contrastStretching.setAutoExclusive(False)
            self.RadioButton_DataAug_contrastStretching.setChecked(False)
            self.RadioButton_DataAug_contrastStretching.setAutoExclusive(True)

            self.RadioButton_DataAug_histogramEq.setEnabled(False)
            self.RadioButton_DataAug_histogramEq.setAutoExclusive(False)
            self.RadioButton_DataAug_histogramEq.setChecked(False)
            self.RadioButton_DataAug_histogramEq.setAutoExclusive(True)

            self.RadioButton_DataAug_adaptiveEq.setEnabled(False)
            self.RadioButton_DataAug_adaptiveEq.setAutoExclusive(False)
            self.RadioButton_DataAug_adaptiveEq.setChecked(False)
            self.RadioButton_DataAug_adaptiveEq.setAutoExclusive(True)


########## third tab
    def textChangeAlpha(self,text):
        self.inputalpha = text
        # if text.isdigit():
        #     self.inputalpha=text
        # else:
        #     self.alphaShouldBeNumber()


    def textChangeGamma(self,text):
        self.inputGamma = text
        # if text.isdigit():
        #     self.inputGamma=text
        # else:
        #     self.GammaShouldBeNumber()

    def wheelScroll(self,ind,oncrollStatus):
        if oncrollStatus=='on_scroll':
            self.horizontalSliderPatch.setValue(ind)
            self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)
        elif oncrollStatus=='onscrollW' or oncrollStatus=='onscroll_3D':
            self.wheelScrollW(ind)
        elif oncrollStatus=='onscrollSS':
            self.wheelScrollSS(ind)
        else:
            pass

    def wheelScrollW(self,ind):
        self.horizontalSliderPatch.setValue(ind)
        self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)

    def wheelScrollSS(self,indSS):
        self.horizontalSliderPatch.setValue(indSS)
        self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)

    def clickList(self,qModelIndex):

        self.chosenLayerName = self.qList[qModelIndex.row()]

    def simpleName(self,inpName):
        if "/" in inpName:
            inpName = inpName.split("/")[0]
            if ":" in inpName:
                inpName = inpName.split(':')[0]
        elif ":" in inpName:
            inpName = inpName.split(":")[0]
            if "/" in inpName:
                inpName = inpName.split('/')[0]

        return inpName

    def show_layer_name(self):
        qList = []

        for i in self.act:
            qList.append(i)

            # if self.act[i].ndim==5 and self.modelDimension=='3D':
            #     self.act[i]=np.transpose(self.act[i],(0,4,1,2,3))
        self.qList = qList

    def sliderValue(self):
        if self.W_F=='w':

            self.chosenWeightNumber=self.horizontalSliderPatch.value()
            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
            self.overlay.show()
            self.wyPlot.setDisabled(True)
            self.newW3D = loadImage_weights_plot_3D(self.matplotlibwidget_static, self.w, self.chosenWeightNumber,
                                                    self.totalWeights, self.totalWeightsSlices)
            self.newW3D.trigger.connect(self.loadEnd2)
            self.newW3D.start()

            # self.matplotlibwidget_static.mpl.weights_plot_3D(self.w, self.chosenWeightNumber, self.totalWeights,self.totalWeightsSlices)
        elif self.W_F=='f':

            if self.modelDimension=='2D':
                self.chosenPatchNumber=self.horizontalSliderPatch.value()
                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
                self.overlay.show()
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_features_plot(self.matplotlibwidget_static, self.chosenPatchNumber)
                self.newf.trigger.connect(self.loadEnd2)
                self.newf.start()
                # self.matplotlibwidget_static.mpl.features_plot(self.chosenPatchNumber)
            elif self.modelDimension == '3D':

                self.chosenPatchNumber = self.horizontalSliderPatch.value()
                self.chosenPatchSliceNumber =self.horizontalSliderSlice.value()
                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
                self.overlay.show()
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_features_plot_3D(self.matplotlibwidget_static, self.chosenPatchNumber,
                                                       self.chosenPatchSliceNumber)
                self.newf.trigger.connect(self.loadEnd2)
                self.newf.start()
                # self.matplotlibwidget_static.mpl.features_plot_3D(self.chosenPatchNumber,self.chosenPatchSliceNumber)
        elif self.W_F=='s':

            self.chosenSSNumber = self.horizontalSliderPatch.value()
            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
            self.overlay.setGeometry(QtCore.QRect(700, 350, 171, 141))
            self.overlay.show()
            self.wyPlot.setDisabled(True)
            self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
            self.newf.trigger.connect(self.loadEnd2)
            self.newf.start()
            # self.matplotlibwidget_static.mpl.subset_selection_plot(self.chosenSSNumber)

        else:
            pass

    def sliderValueSS(self):
        self.chosenSSNumber=self.horizontalSliderSS.value()
        # self.matplotlibwidget_static_2.mpl.subset_selection_plot(self.chosenSSNumber)
        self.matplotlibwidget_static.mpl.subset_selection_plot(self.chosenSSNumber)

    @pyqtSlot()
    def on_wyChooseFile_clicked(self):
        self.openfile_name = QtWidgets.QFileDialog.getOpenFileName(self,'Choose the file','.','H5 files(*.h5)')[0]
        if len(self.openfile_name)==0:
            pass
        else:
            self.horizontalSliderPatch.hide()
            self.horizontalSliderSlice.hide()
            self.labelPatch.hide()
            self.labelSlice.hide()
            self.lcdNumberSlice.hide()
            self.lcdNumberPatch.hide()
            self.matplotlibwidget_static.mpl.fig.clf()

            self.model=load_model(self.openfile_name)
            print()



    @pyqtSlot()
    def on_wyInputData_clicked(self):
        self.inputData_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose the file', '.', 'H5 files(*.h5)')[0]
        if len(self.inputData_name)==0:
            pass
        else:
            if len(self.openfile_name) != 0:
                self.horizontalSliderPatch.hide()
                self.horizontalSliderSlice.hide()
                self.labelPatch.hide()
                self.labelSlice.hide()
                self.lcdNumberSlice.hide()
                self.lcdNumberPatch.hide()
                self.matplotlibwidget_static.mpl.fig.clf()

                self.inputData = h5py.File(self.inputData_name,'r')
                # the number of the input
                for i in self.inputData:
                    if i == 'X_test_p2' or i == 'y_test_p2':
                        self.twoInput = True
                        break

                if self.inputData['X_test'].ndim == 4:
                    self.modelDimension = '2D'
                    X_test = self.inputData['X_test'][:, 2052:2160, :, :]
                    X_test = np.transpose(np.array(X_test), (1, 0, 2, 3))
                    self.subset_selection = X_test

                    if self.twoInput:
                        X_test_p2 = self.inputData['X_test_p2'][:, 2052:2160, :, :]
                        X_test_p2 = np.transpose(np.array(X_test_p2), (1, 0, 2, 3))
                        self.subset_selection_2 = X_test_p2


                elif self.inputData['X_test'].ndim == 5:
                    self.modelDimension = '3D'
                    X_test = self.inputData['X_test'][:, 0:20, :, :, :]
                    X_test = np.transpose(np.array(X_test), (1, 0, 2, 3, 4))
                    self.subset_selection = X_test

                    if self.twoInput:
                        X_test_p2 = self.inputData['X_test_p2'][:, 0:20, :, :, :]
                        X_test_p2 = np.transpose(np.array(X_test_p2), (1, 0, 2, 3, 4))
                        self.subset_selection_2 = X_test_p2

                else:
                    print('the dimension of X_test should be 4 or 5')

                if self.twoInput:
                    self.radioButton_3.show()
                    self.radioButton_4.show()


                plot_model(self.model, 'configGUI/model.png')
                if self.twoInput:
                    self.modelInput = self.model.input[0]
                    self.modelInput2 = self.model.input[1]
                else:
                    self.modelInput = self.model.input

                self.layer_index_name = {}
                for i, layer in enumerate(self.model.layers):
                    self.layer_index_name[layer.name] = i


                for i, layer in enumerate(self.model.input_layers):

                    get_activations = K.function([layer.input, K.learning_phase()],
                                                 [layer.output, ])

                    if i == 0:
                        self.act[layer.name] = get_activations([self.subset_selection, 0])[0]
                    elif i == 1:
                        self.act[layer.name] = get_activations([self.subset_selection_2, 0])[0]
                    else:
                        print('no output of the input layer is created')

                for i, layer in enumerate(self.model.layers):
                    # input_len=layer.input.len()
                    if hasattr(layer.input, "__len__"):
                        if len(layer.input) == 2:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function([layer.input[0], layer.input[1], K.learning_phase()],
                                                         [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                             0])[0]

                        elif len(layer.input) == 3:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], K.learning_phase()], [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                             0])[0]

                        elif len(layer.input) == 4:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], K.learning_phase()],
                                [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                    self.act[inputLayerNameList[3]],
                                                                             0])[0]

                        elif len(layer.input) == 5:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                    self.act[inputLayerNameList[3]],
                                                                    self.act[inputLayerNameList[4]],
                                                                             0])[0]
                        elif len(layer.input) == 6:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],layer.input[5],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                    self.act[inputLayerNameList[3]],
                                                                    self.act[inputLayerNameList[4]],
                                                                    self.act[inputLayerNameList[5]],
                                                                             0])[0]

                        elif len(layer.input) == 7:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],layer.input[5],layer.input[6],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                    self.act[inputLayerNameList[3]],
                                                                    self.act[inputLayerNameList[4]],
                                                                    self.act[inputLayerNameList[5]],
                                                                    self.act[inputLayerNameList[6]],
                                                                             0])[0]
                        elif len(layer.input) == 8:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],layer.input[5],layer.input[6],layer.input[7],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                    self.act[inputLayerNameList[3]],
                                                                    self.act[inputLayerNameList[4]],
                                                                    self.act[inputLayerNameList[5]],
                                                                    self.act[inputLayerNameList[6]],
                                                                    self.act[inputLayerNameList[7]],
                                                                             0])[0]

                        elif len(layer.input) == 9:
                            inputLayerNameList = []
                            for ind_li, layerInput in enumerate(layer.input):
                                inputLayerNameList.append(self.simpleName(layerInput.name))

                            get_activations = K.function(
                                [layer.input[0], layer.input[1], layer.input[2], layer.input[3], layer.input[4],layer.input[5],layer.input[6],layer.input[7],layer.input[8],
                                 K.learning_phase()],
                                [layer.output, ])
                            self.act[layer.name] = get_activations([self.act[inputLayerNameList[0]],
                                                                    self.act[inputLayerNameList[1]],
                                                                    self.act[inputLayerNameList[2]],
                                                                    self.act[inputLayerNameList[3]],
                                                                    self.act[inputLayerNameList[4]],
                                                                    self.act[inputLayerNameList[5]],
                                                                    self.act[inputLayerNameList[6]],
                                                                    self.act[inputLayerNameList[7]],
                                                                    self.act[inputLayerNameList[8]],
                                                                             0])[0]
                        else:
                            print('the number of input is more than 9')

                    else:
                        get_activations = K.function([layer.input, K.learning_phase()], [layer.output, ])
                        inputLayerName = self.simpleName(layer.input.name)
                        self.act[layer.name] = get_activations([self.act[inputLayerName], 0])[0]

                dot = model_to_dot(self.model, show_shapes=False, show_layer_names=True, rankdir='TB')
                if hasattr(self.model, "layers_by_depth"):
                    self.layers_by_depth = self.model.layers_by_depth
                elif hasattr(self.model.model, "layers_by_depth"):
                    self.layers_by_depth = self.model.model.layers_by_depth
                else:
                    print('the model or model.model should contain parameter layers_by_depth')

                maxCol = 0

                for i in range(len(self.layers_by_depth)):

                    for ind, layer in enumerate(self.layers_by_depth[i]):  # the layers in No i layer in the model
                        if maxCol < ind:
                            maxCow = ind

                        if len(layer.weights) == 0:
                            w = 0
                        else:

                            w = layer.weights[0]
                            init = tf.global_variables_initializer()
                            with tf.Session() as sess_i:
                                sess_i.run(init)
                                # print(sess_i.run(w))
                                w = sess_i.run(w)

                        self.weights[layer.name] = w

                if self.modelDimension == '3D':
                    for i in self.weights:
                        # a=self.weights[i]
                        # b=a.ndim
                        if hasattr(self.weights[i],"ndim"):
                            if self.weights[i].ndim==5:
                                self.LayerWeights[i] = np.transpose(self.weights[i], (4, 3, 2, 0, 1))
                        else:
                            self.LayerWeights[i] =self.weights[i]
                elif self.modelDimension == '2D':
                    for i in self.weights:
                        if hasattr(self.weights[i], "ndim"):

                            if self.weights[i].ndim == 4:
                                self.LayerWeights[i] = np.transpose(self.weights[i], (3, 2, 0, 1))
                        else:
                            self.LayerWeights[i] = self.weights[i]
                else:
                    print('the dimesnion of the weights should be 2D or 3D')

                self.show_layer_name()

                self.totalSS = len(self.subset_selection)

                # show the activations' name in the List
                slm = QtWidgets.QStringListModel();
                slm.setStringList(self.qList)
                self.listView.setModel(slm)

            else:
                self.showChooseFileDialog()


    @pyqtSlot()
    def on_wyShowArchitecture_clicked(self):
        # Show the structure of the model and plot the weights
        if len(self.openfile_name) != 0:

            self.canvasStructure = MyMplCanvas()

            self.canvasStructure.loadImage()
            self.graphicscene = QtWidgets.QGraphicsScene()
            self.graphicscene.addWidget(self.canvasStructure)
            self.graphicview = Activeview()
            self.scrollAreaWidgetContents = QtWidgets.QWidget()
            self.maingrids = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
            self.scrollArea.setWidget(self.scrollAreaWidgetContents)
            self.maingrids.addWidget(self.graphicview)
            self.graphicview.setScene(self.graphicscene)
            # self.graphicsView.setScene(self.graphicscene)

        else:
            self.showChooseFileDialog()

    @pyqtSlot()
    def on_wyPlot_clicked(self):
        # self.matplotlibwidget_static_2.hide()

        # Show the structure of the model and plot the weights
        if len(self.openfile_name) != 0:
            if self.radioButton.isChecked()== True :
                if len(self.chosenLayerName) != 0:

                    self.W_F='w'
                    # show the weights
                    if self.modelDimension == '2D':
                        if hasattr(self.LayerWeights[self.chosenLayerName], "ndim"):

                            if self.LayerWeights[self.chosenLayerName].ndim==4:
                                self.lcdNumberPatch.hide()
                                self.lcdNumberSlice.hide()
                                self.horizontalSliderPatch.hide()
                                self.horizontalSliderSlice.hide()
                                self.labelPatch.hide()
                                self.labelSlice.hide()

                                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                                self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                                self.overlay.show()

                                self.matplotlibwidget_static.mpl.getLayersWeights(self.LayerWeights)
                                self.wyPlot.setDisabled(True)
                                self.newW2D = loadImage_weights_plot_2D(self.matplotlibwidget_static,self.chosenLayerName)
                                self.newW2D.trigger.connect(self.loadEnd2)
                                self.newW2D.start()

                                # self.matplotlibwidget_static.mpl.weights_plot_2D(self.chosenLayerName)
                                self.matplotlibwidget_static.show()
                            # elif self.LayerWeights[self.chosenLayerName].ndim==0:
                            #     self.showNoWeights()
                            else:
                                self.showWeightsDimensionError()

                        elif self.LayerWeights[self.chosenLayerName]==0:
                            self.showNoWeights()


                    elif self.modelDimension == '3D':
                        if hasattr(self.LayerWeights[self.chosenLayerName],"ndim"):

                            if self.LayerWeights[self.chosenLayerName].ndim == 5:

                                self.w=self.LayerWeights[self.chosenLayerName]
                                self.totalWeights=self.w.shape[0]
                                # self.totalWeightsSlices=self.w.shape[2]
                                self.horizontalSliderPatch.setMinimum(1)
                                self.horizontalSliderPatch.setMaximum(self.totalWeights)
                                # self.horizontalSliderSlice.setMinimum(1)
                                # self.horizontalSliderSlice.setMaximum(self.totalWeightsSlices)
                                self.chosenWeightNumber=1
                                self.horizontalSliderPatch.setValue(self.chosenWeightNumber)

                                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                                self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                                self.overlay.show()

                                self.wyPlot.setDisabled(True)
                                self.newW3D = loadImage_weights_plot_3D(self.matplotlibwidget_static, self.w,self.chosenWeightNumber,self.totalWeights,self.totalWeightsSlices)
                                self.newW3D.trigger.connect(self.loadEnd2)
                                self.newW3D.start()

                                # self.matplotlibwidget_static.mpl.weights_plot_3D(self.w,self.chosenWeightNumber,self.totalWeights,self.totalWeightsSlices)

                                self.matplotlibwidget_static.show()
                                self.horizontalSliderSlice.hide()
                                self.horizontalSliderPatch.show()
                                self.labelPatch.show()
                                self.labelSlice.hide()
                                self.lcdNumberSlice.hide()
                                self.lcdNumberPatch.show()
                            # elif self.LayerWeights[self.chosenLayerName].ndim==0:
                            #     self.showNoWeights()
                            else:
                                self.showWeightsDimensionError3D()

                        elif self.LayerWeights[self.chosenLayerName]==0:
                            self.showNoWeights()

                    else:
                        print('the dimesnion should be 2D or 3D')

                else:
                    self.showChooseLayerDialog()

            elif self.radioButton_2.isChecked()== True :
                if len(self.chosenLayerName) != 0:
                    self.W_F = 'f'
                    if self.modelDimension == '2D':
                        if self.act[self.chosenLayerName].ndim==4:
                            self.activations=self.act[self.chosenLayerName]
                            self.totalPatches=self.activations.shape[0]

                            self.matplotlibwidget_static.mpl.getLayersFeatures(self.activations, self.totalPatches)

                            # show the features
                            self.chosenPatchNumber=1
                            self.horizontalSliderPatch.setMinimum(1)
                            self.horizontalSliderPatch.setMaximum(self.totalPatches)
                            self.horizontalSliderPatch.setValue(self.chosenPatchNumber)

                            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                            self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                            self.overlay.show()
                            self.wyPlot.setDisabled(True)
                            self.newf = loadImage_features_plot(self.matplotlibwidget_static,self.chosenPatchNumber)
                            self.newf.trigger.connect(self.loadEnd2)
                            self.newf.start()

                            # self.matplotlibwidget_static.mpl.features_plot(self.chosenPatchNumber)
                            self.matplotlibwidget_static.show()
                            self.horizontalSliderSlice.hide()
                            self.horizontalSliderPatch.show()
                            self.labelPatch.show()
                            self.labelSlice.hide()
                            self.lcdNumberPatch.show()
                            self.lcdNumberSlice.hide()
                        else:
                            self.showNoFeatures()

                    elif self.modelDimension =='3D':
                        a=self.act[self.chosenLayerName]
                        if self.act[self.chosenLayerName].ndim == 5:
                            self.activations = self.act[self.chosenLayerName]
                            self.totalPatches = self.activations.shape[0]
                            self.totalPatchesSlices=self.activations.shape[1]

                            self.matplotlibwidget_static.mpl.getLayersFeatures_3D(self.activations, self.totalPatches,self.totalPatchesSlices)

                            self.chosenPatchNumber=1
                            self.chosenPatchSliceNumber=1
                            self.horizontalSliderPatch.setMinimum(1)
                            self.horizontalSliderPatch.setMaximum(self.totalPatches)
                            self.horizontalSliderPatch.setValue(self.chosenPatchNumber)
                            self.horizontalSliderSlice.setMinimum(1)
                            self.horizontalSliderSlice.setMaximum(self.totalPatchesSlices)
                            self.horizontalSliderSlice.setValue(self.chosenPatchSliceNumber)

                            self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                            self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                            self.overlay.show()
                            self.wyPlot.setDisabled(True)
                            self.newf = loadImage_features_plot_3D(self.matplotlibwidget_static, self.chosenPatchNumber,self.chosenPatchSliceNumber)
                            self.newf.trigger.connect(self.loadEnd2)
                            self.newf.start()

                            # self.matplotlibwidget_static.mpl.features_plot_3D(self.chosenPatchNumber,self.chosenPatchSliceNumber)
                            self.horizontalSliderSlice.show()
                            self.horizontalSliderPatch.show()
                            self.labelPatch.show()
                            self.labelSlice.show()
                            self.lcdNumberPatch.show()
                            self.lcdNumberSlice.show()
                            self.matplotlibwidget_static.show()
                        else:
                            self.showNoFeatures()

                    else:
                        print('the dimesnion should be 2D or 3D')

                else:
                    self.showChooseLayerDialog()

            else:
                self.showChooseButtonDialog()

        else:
            self.showChooseFileDialog()

    @pyqtSlot()
    def on_wySubsetSelection_clicked(self):
        # Show the Subset Selection
        if len(self.openfile_name) != 0:
            # show the weights
            # self.scrollArea.hide()
            self.W_F ='s'
            self.chosenSSNumber = 1
            self.horizontalSliderPatch.setMinimum(1)
            self.horizontalSliderPatch.setMaximum(self.totalSS)
            self.horizontalSliderPatch.setValue(self.chosenSSNumber)
            self.horizontalSliderPatch.valueChanged.connect(self.lcdNumberPatch.display)
            self.lcdNumberPatch.show()
            self.lcdNumberSlice.hide()
            self.horizontalSliderPatch.show()
            self.horizontalSliderSlice.hide()
            self.labelPatch.show()
            self.labelSlice.hide()


            # create input patch
            if self.twoInput==False:
                self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection, self.totalSS)

                self.createSubset(self.modelInput,self.subset_selection)
                self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                self.overlay.show()
                self.wyPlot.setDisabled(True)
                self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                self.newf.trigger.connect(self.loadEnd2)
                self.newf.start()

                # self.matplotlibwidget_static.mpl.subset_selection_plot(self.chosenSSNumber)
            elif self.twoInput:
                if self.radioButton_3.isChecked(): # the 1st input
                    self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection, self.totalSS)
                    self.createSubset(self.modelInput,self.subset_selection)
                    self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                    self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                    self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                    self.overlay.show()
                    self.wyPlot.setDisabled(True)
                    self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                    self.newf.trigger.connect(self.loadEnd2)
                    self.newf.start()

                elif self.radioButton_4.isChecked(): # the 2nd input
                    self.matplotlibwidget_static.mpl.getSubsetSelections(self.subset_selection_2, self.totalSS)
                    self.createSubset(self.modelInput2,self.subset_selection_2)
                    self.matplotlibwidget_static.mpl.getSSResult(self.ssResult)

                    self.overlay = Overlay(self.centralWidget())  # self.scrollArea self.centralWidget()
                    self.overlay.setGeometry(QtCore.QRect(500, 350, 171, 141))
                    self.overlay.show()
                    self.wyPlot.setDisabled(True)
                    self.newf = loadImage_subset_selection_plot(self.matplotlibwidget_static, self.chosenSSNumber)
                    self.newf.trigger.connect(self.loadEnd2)
                    self.newf.start()

                else:
                    self.showChooseInput()
            else:
                print('the number of input should be 1 or 2')

        else:
            self.showChooseFileDialog()

    def clickList_1(self, qModelIndex):
        self.chosenActivationName = self.qList[qModelIndex.row()]

    def showChooseFileDialog(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "Please select one H5 File at first",
                                                  QtWidgets.QMessageBox.Ok )

    def showChooseLayerDialog(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "Please select one Layer at first",
                                                  QtWidgets.QMessageBox.Ok)

    def showChooseButtonDialog(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "Please select to plot the weights or the features",
                                                  QtWidgets.QMessageBox.Ok)

    def showNoWeights(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "This layer does not have weighst,please select other layers",
                                                  QtWidgets.QMessageBox.Ok)

    def showWeightsDimensionError(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "The diemnsion of the weights should be 0 or 4",
                                                  QtWidgets.QMessageBox.Ok)

    def showWeightsDimensionError3D(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "The diemnsion of the weights should be 0 or 5",
                                                  QtWidgets.QMessageBox.Ok)

    def showNoFeatures(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "This layer does not have feature maps, please select other layers",
                                                  QtWidgets.QMessageBox.Ok)

    def loadEnd2(self):
        self.overlay.killTimer(self.overlay.timer)
        self.overlay.hide()
        self.wyPlot.setDisabled(False)

    def alphaShouldBeNumber(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "Alpha should be a number!!!",
                                                  QtWidgets.QMessageBox.Ok)

    def GammaShouldBeNumber(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "Gamma should be a number!!!",
                                                  QtWidgets.QMessageBox.Ok)

    def createSubset(self,modelInput,subset_selection):
        class_idx = 0
        reg_param = 1 / (2e-4)

        input = modelInput  # tensor
        cost = -K.sum(K.log(input[:, class_idx] + 1e-8))  # tensor
        gradient = K.gradients(cost, input)  # list

        sess = tf.InteractiveSession()
        calcCost = network_visualization.TensorFlowTheanoFunction([input], cost)
        calcGrad = network_visualization.TensorFlowTheanoFunction([input], gradient)

        step_size = float(self.inputalpha)
        reg_param = float(self.inputGamma)

        test = subset_selection
        data_c = test
        oss_v = network_visualization.SubsetSelection(calcGrad, calcCost, data_c, alpha=reg_param, gamma=step_size)
        result = oss_v.optimize(np.random.uniform(0, 1.0, size=data_c.shape))
        result = result * test
        result[result>0]=1
        self.ssResult=result

    def showChooseInput(self):
        reply = QtWidgets.QMessageBox.information(self,
                                        "Warning",
                                        "Please select to plot the input 1 or 2",
                                        QtWidgets.QMessageBox.Ok)

    def select_roi_OnOff(self):

        if self.gridson:
            with open('configGUI/lastWorkspace.json', 'r') as json_data:
                lastState = json.load(json_data)
                lastState['mode'] = self.gridsnr  ###
                if self.gridsnr == 2:
                    lastState['layout'][0] = self.layoutlines
                    lastState['layout'][1] = self.layoutcolumns
                else:
                    lastState['layout'][0] = self.layout3D

                global pathlist, list1, pnamelist, problist, hatchlist, correslist, cnrlist, shapelist, indlist, ind2list, ind3list
                # shapelist = (list(shapelist)).tolist()
                # shapelist = pd.Series(shapelist).to_json(orient='values')
                lastState['Shape'] = shapelist
                lastState['Pathes'] = pathlist
                lastState['NResults'] = pnamelist
                lastState['NrClass'] = cnrlist
                lastState['Corres'] = correslist
                lastState['Index'] = indlist
                lastState['Index2'] = ind2list
                lastState['Index3'] = ind3list

            with open('configGUI/lastWorkspace.json', 'w') as json_data:
                json_data.write(json.dumps(lastState))

            listA = open('config/dump1.txt', 'wb')
            pickle.dump(list1, listA)
            listA.close()
            listB = open('config/dump2.txt', 'wb')
            pickle.dump(problist, listB)
            listB.close()
            listC = open('config/dump3.txt', 'wb')
            pickle.dump(hatchlist, listC)
            listC.close()

        subprocess.Popen(["python", 'configGUI/ROI_Selector.py'])

    def mouse_tracking(self):

        if self.gridson:
            with open('configGUI/lastWorkspace.json', 'r') as json_data:
                lastState = json.load(json_data)
                lastState['mode'] = self.gridsnr  ###
                if self.gridsnr == 2:
                    lastState['layout'][0] = self.layoutlines
                    lastState['layout'][1] = self.layoutcolumns
                else:
                    lastState['layout'][0] = self.layout3D

                global pathlist, list1, pnamelist, problist, hatchlist, correslist, cnrlist, shapelist, indlist, ind2list, ind3list
                # shapelist = (list(shapelist)).tolist()
                # shapelist = pd.Series(shapelist).to_json(orient='values')
                lastState['Shape'] = shapelist
                lastState['Pathes'] = pathlist
                lastState['NResults'] = pnamelist
                lastState['NrClass'] = cnrlist
                lastState['Corres'] = correslist
                lastState['Index'] = indlist
                lastState['Index2'] = ind2list
                lastState['Index3'] = ind3list

            with open('configGUI/lastWorkspace.json', 'w') as json_data:
                json_data.write(json.dumps(lastState))

            listA = open('config/dump1.txt', 'wb')
            pickle.dump(list1, listA)
            listA.close()
            listB = open('config/dump2.txt', 'wb')
            pickle.dump(problist, listB)
            listB.close()
            listC = open('config/dump3.txt', 'wb')
            pickle.dump(hatchlist, listC)
            listC.close()

        subprocess.Popen(["python", 'configGUI/mouse_tracking'])



############################################################ class of grids
class Viewgroup(QtWidgets.QGridLayout):

    in_link = QtCore.pyqtSignal()
    rotateSignal = QtCore.pyqtSignal(str)
    scrollSignal = QtCore.pyqtSignal(str)
    zoomSignal = QtCore.pyqtSignal(str)

    def __init__(self, parent: object = None) -> object:
        super(Viewgroup, self).__init__(parent)

        self.mode = 1
        self.viewnr1 = 1
        self.viewnr2 = 1
        self.spin = None
        self.anewcanvas = None
        self.islinked = False
        self.skiplink = False
        self.skipdis = True
        self.labeled = False

        self.pathbox = QtWidgets.QComboBox()
        self.pathbox.addItem('closed')
        self.addWidget(self.pathbox, 0, 0, 1, 1)
        self.refbox = QtWidgets.QComboBox()
        self.refbox.addItem('closed')
        self.addWidget(self.refbox, 0, 1, 1, 1)
        self.pathbox.setDisabled(False)
        self.refbox.setDisabled(True)

        self.graylabel = QtWidgets.QLabel()
        self.zoomlabel = QtWidgets.QLabel()
        self.slicelabel = QtWidgets.QLabel()
        self.graylabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.graylabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zoomlabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.zoomlabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slicelabel.setFrameShape(QtWidgets.QFrame.Panel)
        self.slicelabel.setFrameShadow(QtWidgets.QFrame.Raised)

        self.secondline = QtWidgets.QGridLayout()
        self.secondline.addWidget(self.slicelabel, 0, 0, 1, 1)
        self.secondline.addWidget(self.zoomlabel, 0, 1, 1, 1)
        self.secondline.addWidget(self.graylabel, 0, 2, 1, 1)
        self.addLayout(self.secondline, 1, 0, 1, 2)

        self.modechange = QtWidgets.QPushButton()
        self.addWidget(self.modechange, 0, 2, 1, 1)
        self.imrotate = QtWidgets.QPushButton()
        self.addWidget(self.imrotate, 1, 3, 1, 1)
        self.imageedit = QtWidgets.QPushButton()
        self.addWidget(self.imageedit, 1, 2, 1, 1)
        self.linkon = QtWidgets.QPushButton()
        self.addWidget(self.linkon, 0, 3, 1, 1)

        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/Icons/switch2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon2 = QtGui.QIcon()
        self.icon2.addPixmap(QtGui.QPixmap(":/icons/Icons/blink.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/Icons/edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/Icons/rotate.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon5 = QtGui.QIcon()
        self.icon5.addPixmap(QtGui.QPixmap(":/icons/Icons/link.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.sizePolicy.setHorizontalStretch(0)
        self.sizePolicy.setVerticalStretch(0)
        self.sizePolicy.setHeightForWidth(self.modechange.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.linkon.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.imageedit.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.imrotate.sizePolicy().hasHeightForWidth())
        self.modechange.setSizePolicy(self.sizePolicy)
        self.modechange.setText("")
        self.modechange.setIcon(icon1)
        self.modechange.setToolTip("change source between path and reference")
        self.modechange.setDisabled(True)
        self.linkon.setSizePolicy(self.sizePolicy)
        self.linkon.setText("")
        self.linkon.setIcon(self.icon2)
        self.linkon.setToolTip("link on between 3 windows")
        self.imageedit.setSizePolicy(self.sizePolicy)
        self.imageedit.setText("")
        self.imageedit.setIcon(icon3)
        self.imageedit.setToolTip("edit gray scale")
        self.imrotate.setSizePolicy(self.sizePolicy)
        self.imrotate.setText("")
        self.imrotate.setIcon(icon4)
        self.imrotate.setToolTip("change view_image XY/YZ/XZ\nduring labeling\nafter change view_image\nplease click_event button\nnoselection on")
        self.Viewpanel = Activeview()
        self.addWidget(self.Viewpanel, 2, 0, 1, 4)

        self.anewscene = Activescene()
        self.Viewpanel.setScene(self.anewscene)
        self.Viewpanel.zooming_data.connect(self.updateZoom)

        self.modechange.clicked.connect(self.switchMode)
        self.imrotate.clicked.connect(self.rotateView)
        self.pathbox.currentIndexChanged.connect(self.loadScene)
        self.refbox.currentIndexChanged.connect(self.loadScene)
        self.imageedit.clicked.connect(self.setGrey)
        self.linkon.clicked.connect(self.linkPanel)
        self.oldindex = 0

        self.zoomscale = 1.0

    def switchMode(self):
        if self.mode ==1:
            self.mode = 2
            self.pathbox.setDisabled(True)
            self.refbox.setDisabled(False)
            self.pathbox.setCurrentIndex(0)
        else:
            self.mode = 1
            self.pathbox.setDisabled(False)
            self.refbox.setDisabled(True)
            self.refbox.setCurrentIndex(0)

    def rotateView(self):
        if self.pathbox.currentIndex() != 0:
            if self.mode == 1:
                if self.viewnr1 == 1:
                    self.viewnr1 = 2
                    param = {'image': list1[self.spin - 1], 'mode': 2, 'shape':shapelist[self.spin - 1]}
                    self.anewcanvas = Canvas(param)
                elif self.viewnr1 == 2:
                    self.viewnr1 = 3
                    param = {'image': list1[self.spin - 1], 'mode': 3, 'shape':shapelist[self.spin - 1]}
                    self.anewcanvas = Canvas(param)
                elif self.viewnr1 == 3:
                    self.viewnr1 = 1
                    param = {'image': list1[self.spin - 1], 'mode': 1, 'shape':shapelist[self.spin - 1]}
                    self.anewcanvas = Canvas(param)
                self.loadImage()

            else:

                if cnrlist[self.spin - 1] == 11:
                    if self.viewnr2 == 1:
                        self.viewnr2 = 2
                        param2 = {'image': list1[correslist[self.spin - 1]], 'mode': 5, 'color': problist[self.spin - 1],
                                  'hatch': hatchlist[self.spin - 1], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                                  'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param2)
                    elif self.viewnr2 == 2:
                        self.viewnr2 = 3
                        param3 = {'image': list1[correslist[self.spin - 1]], 'mode': 6, 'color': problist[self.spin - 1],
                                  'hatch': hatchlist[self.spin - 1], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                                  'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param3)
                    elif self.viewnr2 == 3:
                        self.viewnr2 = 1
                        param1 = {'image': list1[correslist[self.spin - 1]], 'mode': 4, 'color': problist[self.spin - 1],
                                  'hatch': hatchlist[self.spin - 1], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                                  'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param1)

                elif cnrlist[self.spin - 1] == 2:
                    if self.viewnr2 == 1:
                        self.viewnr2 = 2
                        param2 = {'image': list1[correslist[self.spin - 1]], 'mode': 8, 'color': problist[self.spin - 1],
                                  'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param2)
                    elif self.viewnr2 == 2:
                        self.viewnr2 = 3
                        param3 = {'image': list1[correslist[self.spin - 1]], 'mode': 9, 'color': problist[self.spin - 1],
                                  'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param3)
                    elif self.viewnr2 == 3:
                        self.viewnr2 = 1
                        param1 = {'image': list1[correslist[self.spin - 1]], 'mode': 7, 'color': problist[self.spin - 1],
                                  'cmap': cmap1, 'trans': vtr1, 'shape': shapelist[correslist[self.spin - 1]]}
                        self.anewcanvas = Canvas(param1)

                self.loadImage()
            self.rotateSignal.emit(self.current_slice())

        else:
            pass

    def linkPanel(self):
        if self.pathbox.currentIndex() == 0 and self.refbox.currentIndex() == 0: # trigger condition
            pass
        else:
            if self.islinked == False:
                self.linkon.setIcon(self.icon5)
                self.islinked = True
                self.in_link.emit()
            else:
                self.linkon.setIcon(self.icon2)
                self.islinked = False
                self.in_link.emit()

    def loadImage(self):
        self.anewscene.clear()
        self.anewscene.addWidget(self.anewcanvas)
        self.Viewpanel.zoomback()
        self.slicelabel.setText('S %s' % (self.anewcanvas.ind + 1) + '/ %s' % (self.anewcanvas.slices))
        self.graylabel.setText('G %s' % (self.anewcanvas.graylist))
        if self.viewnr1==2:
            self.zoomlabel.setText('YZ')
        elif self.viewnr1==3:
            self.zoomlabel.setText('XZ')
        elif self.viewnr1==1:
            self.zoomlabel.setText('XY')

        self.anewcanvas.update_data.connect(self.updateSlices)
        self.anewcanvas.gray_data.connect(self.updateGray)
        self.anewcanvas.new_page.connect(self.newSliceview)
        self.anewcanvas.mpl_connect('motion_notify_event', self.mouse_move)

        self.currentImage = self.pathbox.currentText()

        if self.islinked == True:
            self.Viewpanel.zoom_link.disconnect()
            self.Viewpanel.move_link.disconnect()
            self.skiplink = False
            self.skipdis = True
            self.in_link.emit()

    def current_image(self):
        if not self.currentImage.find('_Labeling') == -1:
            self.currentImage = self.currentImage.replace('_Labeling', '')
        return self.currentImage

    def current_slice(self):
        if self.viewnr1==2:
            slice = 'X %s' % (self.anewcanvas.ind + 1)
        elif self.viewnr1==3:
            slice = 'Y %s' % (self.anewcanvas.ind + 1)
        elif self.viewnr1==1:
            slice = 'Z %s' % (self.anewcanvas.ind + 1)
        return slice

    def setlinkoff(self, value):
        self.linkon.setDisabled(value)

    def addPathd(self, pathDicom, value):
        imageItems = [self.pathbox.itemText(i) for i in range(self.pathbox.count())]
        region = os.path.split(pathDicom)
        proband = os.path.split(os.path.split(region[0])[0])[1]
        region = region[1]
        if value == 0:
            newItem = '[' + (proband) + '][' + (region)+ ']'
        else:
            newItem = '[' + (proband) + '][' + (region) + ']_Labeling'
        if newItem not in imageItems:
           self.pathbox.addItem(newItem)

    def addPathre(self, pathColor):
        self.refbox.addItem(pathColor)

    def loadScene(self, i):
        self.spin = i # position in combobox
        self.viewnr1 = 1
        self.viewnr2 = 1
        if i != 0:
            if self.mode == 1:
                param = {'image':list1[i - 1], 'mode':1, 'shape':shapelist[i - 1]}
                self.anewcanvas = Canvas(param)
            else:
                if cnrlist[i - 1] == 11:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 4, 'color': problist[i - 1],
                              'hatch': hatchlist[i - 1], 'cmap': cmap3, 'hmap': hmap2, 'trans': vtr3,
                              'shape': shapelist[correslist[i - 1]]}
                    self.anewcanvas = Canvas(param1)
                elif cnrlist[i - 1] == 2:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 7 , 'color':problist[i - 1],
                               'cmap':cmap1, 'trans':vtr1, 'shape': shapelist[correslist[i - 1]]}
                    self.anewcanvas = Canvas(param1)

            self.loadImage()
        else:
            if self.oldindex != 0:
                self.anewscene.clear()
                self.slicelabel.setText('')
                self.graylabel.setText('')
                self.zoomlabel.setText('')
                if self.islinked == True:
                    self.linkon.setIcon(self.icon2)
                    self.islinked = False
                    self.skiplink = False
                    self.skipdis = True
                    self.Viewpanel.zoom_link.disconnect()
                    self.Viewpanel.move_link.disconnect()
                    self.in_link.emit()
            else:
                pass

        self.oldindex = i

    def newSliceview(self):
        self.graylabel.setText('G %s' % (self.anewcanvas.graylist))

    def updateSlices(self, elist):
        self.slicelabel.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))
        indlist.append(elist[0])
        self.scrollSignal.emit(self.current_slice())

    def updateZoom(self, data):
        self.zoomscale = data
        if self.viewnr1==2:
            self.zoomlabel.setText('YZ %s' % (data))
        elif self.viewnr1==3:
            self.zoomlabel.setText('XZ %s' % (data))
        elif self.viewnr1==1:
            self.zoomlabel.setText('XY %s' % (data))
        self.zoomSignal.emit(self.zoomlabel.text())

    def updateGray(self, elist):
        self.graylabel.setText('G %s' % (elist))

    def clearWidgets(self):
        for i in reversed(range(self.count())):
            widget = self.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def setGrey(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.anewcanvas:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.anewcanvas.set_greyscale(greylist)

    def mouse_move(self, event):
        x = event.xdata
        y = event.ydata
        if y:
            z = y/3.3
            if x:
                x = "%.2f" % x
                y = "%.2f" % y
                z = "%.2f" % z
                if self.viewnr1==1:
                    self.zoomlabel.setText('XY %s' % self.zoomscale + "             Pos     " + 'X %s' % x + "      " + 'Y %s' % y)
                elif self.viewnr1==2:
                    y = x
                    self.zoomlabel.setText('YZ %s' % self.zoomscale + "             Pos     " + "Y %s" % y + "      " + 'Z %s' % z)
                elif self.viewnr1==3:
                    self.zoomlabel.setText('XZ %s' % self.zoomscale + "             Pos     " + "X %s" % x + "      " + 'Z %s' % z)
                else:
                    pass
        else:
            pass

class Viewline(QtWidgets.QGridLayout):
    in_link = QtCore.pyqtSignal()
    def __init__(self, parent=None):
        super(Viewline, self).__init__(parent)

        self.vmode = 1
        self.oldindex = 0
        self.newcanvas1 = None
        self.newcanvas2 = None
        self.newcanvas3 = None
        self.islinked = False
        self.skiplink = False
        self.skipdis = True

        self.gridLayout_1 = QtWidgets.QGridLayout()
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_4 = QtWidgets.QGridLayout()

        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/Icons/switch2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon2 = QtGui.QIcon()
        self.icon2.addPixmap(QtGui.QPixmap(":/icons/Icons/blink.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/Icons/edit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.icon4 = QtGui.QIcon()
        self.icon4.addPixmap(QtGui.QPixmap(":/icons/Icons/link.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.sizePolicy.setHorizontalStretch(0)
        self.sizePolicy.setVerticalStretch(0)

        self.imagelist = QtWidgets.QComboBox()
        self.imagelist.addItem('closed')
        self.reflist = QtWidgets.QComboBox()
        self.reflist.addItem('closed')
        self.reflist.setDisabled(True)
        self.bimre = QtWidgets.QPushButton()
        self.bimre.setText('')
        self.bimre.setSizePolicy(self.sizePolicy)
        self.bimre.setIcon(icon1)
        self.blinkon = QtWidgets.QPushButton()
        self.blinkon.setText('')
        self.blinkon.setSizePolicy(self.sizePolicy)
        self.blinkon.setIcon(self.icon2)
        self.gridLayout_1.addWidget(self.imagelist, 0, 0, 1, 1)
        self.gridLayout_1.addWidget(self.reflist, 0, 1, 1, 1)
        self.gridLayout_1.addWidget(self.bimre, 0, 2, 1, 1)
        self.gridLayout_1.addWidget(self.blinkon, 0, 3, 1, 1)

        self.ed31 = QtWidgets.QPushButton()
        self.ed31.setText('')
        self.ed31.setSizePolicy(self.sizePolicy)
        self.ed31.setIcon(icon3)
        self.ed32 = QtWidgets.QPushButton()
        self.ed32.setText('')
        self.ed32.setSizePolicy(self.sizePolicy)
        self.ed32.setIcon(icon3)
        self.ed33 = QtWidgets.QPushButton()
        self.ed33.setText('')
        self.ed33.setSizePolicy(self.sizePolicy)
        self.ed33.setIcon(icon3)
        self.sizePolicy.setHeightForWidth(self.bimre.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.blinkon.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.ed31.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.ed32.sizePolicy().hasHeightForWidth())
        self.sizePolicy.setHeightForWidth(self.ed33.sizePolicy().hasHeightForWidth())

        self.grt1 = QtWidgets.QLabel()
        self.zot1 = QtWidgets.QLabel()
        self.grt2 = QtWidgets.QLabel()
        self.zot2 = QtWidgets.QLabel()
        self.grt3 = QtWidgets.QLabel()
        self.zot3 = QtWidgets.QLabel()
        self.grt1.setFrameShape(QtWidgets.QFrame.Panel)
        self.grt1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zot1.setFrameShape(QtWidgets.QFrame.Panel)
        self.zot1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.grt2.setFrameShape(QtWidgets.QFrame.Panel)
        self.grt2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zot2.setFrameShape(QtWidgets.QFrame.Panel)
        self.zot2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.grt3.setFrameShape(QtWidgets.QFrame.Panel)
        self.grt3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.zot3.setFrameShape(QtWidgets.QFrame.Panel)
        self.zot3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slt1 = QtWidgets.QLabel()
        self.slt2 = QtWidgets.QLabel()
        self.slt3 = QtWidgets.QLabel()
        self.slt1.setFrameShape(QtWidgets.QFrame.Panel)
        self.slt1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slt2.setFrameShape(QtWidgets.QFrame.Panel)
        self.slt2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.slt3.setFrameShape(QtWidgets.QFrame.Panel)
        self.slt3.setFrameShadow(QtWidgets.QFrame.Raised)

        self.gridLayout_2.addWidget(self.slt1, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.zot1, 1, 1, 1, 1)
        self.gridLayout_2.addWidget(self.grt1, 1, 2, 1, 1)
        self.gridLayout_2.addWidget(self.ed31, 1, 3, 1, 1)

        self.gridLayout_3.addWidget(self.slt2, 1, 0, 1, 1)
        self.gridLayout_3.addWidget(self.zot2, 1, 1, 1, 1)
        self.gridLayout_3.addWidget(self.grt2, 1, 2, 1, 1)
        self.gridLayout_3.addWidget(self.ed32, 1, 3, 1, 1)

        self.gridLayout_4.addWidget(self.slt3, 1, 0, 1, 1)
        self.gridLayout_4.addWidget(self.zot3, 1, 1, 1, 1)
        self.gridLayout_4.addWidget(self.grt3, 1, 2, 1, 1)
        self.gridLayout_4.addWidget(self.ed33, 1, 3, 1, 1)

        self.addLayout(self.gridLayout_1, 0, 0, 1, 3)
        self.addLayout(self.gridLayout_2, 1, 0, 1, 1)
        self.addLayout(self.gridLayout_3, 1, 1, 1, 1)
        self.addLayout(self.gridLayout_4, 1, 2, 1, 1)

        self.Viewpanel1 = Activeview()
        self.Viewpanel2 = Activeview()
        self.Viewpanel3 = Activeview()
        self.scene1 = Activescene()
        self.scene2 = Activescene()
        self.scene3 = Activescene()

        self.Viewpanel1.zooming_data.connect(self.updateZoom1)
        self.Viewpanel2.zooming_data.connect(self.updateZoom2)
        self.Viewpanel3.zooming_data.connect(self.updateZoom3)

        self.addWidget(self.Viewpanel1, 2, 0, 1, 1)
        self.addWidget(self.Viewpanel2, 2, 1, 1, 1)
        self.addWidget(self.Viewpanel3, 2, 2, 1, 1)
        self.itemAtPosition(2, 0).widget().setScene(self.scene1)
        self.itemAtPosition(2, 1).widget().setScene(self.scene2)
        self.itemAtPosition(2, 2).widget().setScene(self.scene3)

        self.bimre.clicked.connect(self.switchMode)
        self.imagelist.currentIndexChanged.connect(self.loadScene)
        self.ed31.clicked.connect(self.setGrey1)
        self.ed32.clicked.connect(self.setGrey2)
        self.ed33.clicked.connect(self.setGrey3)
        self.reflist.currentIndexChanged.connect(self.loadScene)
        self.blinkon.clicked.connect(self.linkPanel)

    def switchMode(self):
        if self.vmode == 1:
            self.vmode = 2
            # self.im_re.setText('Result')
            self.reflist.setDisabled(False)
            self.imagelist.setDisabled(True)
            self.imagelist.setCurrentIndex(0)
            self.reflist.setCurrentIndex(0)
        else:
            self.vmode = 1
            # self.im_re.setText('Image')
            self.reflist.setDisabled(True)
            self.imagelist.setDisabled(False)
            self.reflist.setCurrentIndex(0)
            self.imagelist.setCurrentIndex(0)

    def addPathim(self, pathDicom):
        imageItems = [self.imagelist.itemText(i) for i in range(self.imagelist.count())]
        region = os.path.split(pathDicom)
        proband = os.path.split(os.path.split(region[0])[0])[1]
        region = region[1]
        newItem = 'Proband: %s' % (proband) + '   Image: %s' % (region)
        if newItem not in imageItems:
             self.imagelist.addItem(newItem)

    def addPathre(self, pathColor):
        self.reflist.addItem(pathColor)

    def linkPanel(self):
        if self.imagelist.currentIndex() == 0 and self.reflist.currentIndex() ==0: # trigger condition
            pass
        else:
            if self.islinked == False:
                self.blinkon.setIcon(self.icon4)
                self.islinked = True
                self.in_link.emit()
            else:
                self.blinkon.setIcon(self.icon2)
                self.islinked = False
                self.in_link.emit()

    def loadScene(self, i):
        if i != 0:
            if self.vmode == 1:
                param1 = {'image': list1[i - 1], 'mode': 1, 'shape': shapelist[i - 1]}
                param2 = {'image': list1[i - 1], 'mode': 2, 'shape': shapelist[i - 1]}
                param3 = {'image': list1[i - 1], 'mode': 3, 'shape': shapelist[i - 1]}
                self.newcanvas1 = Canvas(param1)
                self.newcanvas2 = Canvas(param2)
                self.newcanvas3 = Canvas(param3)
            else:
                if cnrlist[i - 1] == 11:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 4 , 'color':problist[i - 1],
                              'hatch':hatchlist[i - 1], 'cmap':cmap3, 'hmap':hmap2, 'trans':vtr3,
                              'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas1 = Canvas(param1)
                    param2 = {'image': list1[correslist[i - 1]], 'mode': 5 , 'color':problist[i - 1],
                              'hatch':hatchlist[i - 1], 'cmap':cmap3, 'hmap':hmap2, 'trans':vtr3,
                              'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas2 = Canvas(param2)
                    param3 = {'image': list1[correslist[i - 1]], 'mode': 6 , 'color':problist[i - 1],
                              'hatch':hatchlist[i - 1], 'cmap':cmap3, 'hmap':hmap2, 'trans':vtr3,
                              'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas3 = Canvas(param3)
                elif cnrlist[i - 1] == 2:
                    param1 = {'image': list1[correslist[i - 1]], 'mode': 7 , 'color':problist[i - 1],
                               'cmap':cmap1, 'trans':vtr1, 'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas1 = Canvas(param1)
                    param2 = {'image': list1[correslist[i - 1]], 'mode': 8 , 'color':problist[i - 1],
                             'cmap':cmap1, 'trans':vtr1, 'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas2 = Canvas(param2)
                    param3 = {'image': list1[correslist[i - 1]], 'mode': 9 , 'color':problist[i - 1],
                               'cmap':cmap1, 'trans':vtr1, 'shape': shapelist[correslist[i - 1]]}
                    self.newcanvas3 = Canvas(param3)
                # elif:
            self.loadImage()
        else:
            if self.oldindex != 0:
                self.scene1.clear()
                self.scene2.clear()
                self.scene3.clear()
                self.slt1.setText('')
                self.grt1.setText('')
                self.zot1.setText('')
                self.slt2.setText('')
                self.grt2.setText('')
                self.zot2.setText('')
                self.slt3.setText('')
                self.grt3.setText('')
                self.zot3.setText('')
                if self.islinked == True:
                    self.islinked = False
                    self.skiplink = False
                    self.skipdis = True
                    self.Viewpanel1.zoom_link.disconnect()
                    self.Viewpanel1.move_link.disconnect()
                    self.Viewpanel2.zoom_link.disconnect()
                    self.Viewpanel2.move_link.disconnect()
                    self.Viewpanel3.zoom_link.disconnect()
                    self.Viewpanel3.move_link.disconnect()
                    self.in_link.emit()
            else:
                pass

        self.oldindex = i

    def loadImage(self):
        self.scene1.clear()
        self.scene2.clear()
        self.scene3.clear()
        self.scene1.addWidget(self.newcanvas1)
        self.scene2.addWidget(self.newcanvas2)
        self.scene3.addWidget(self.newcanvas3)
        self.Viewpanel1.zoomback()
        self.Viewpanel2.zoomback()
        self.Viewpanel3.zoomback()
        self.slt1.setText('S %s' % (self.newcanvas1.ind + 1) + '/ %s' % (self.newcanvas1.slices))
        self.grt1.setText('G %s' % (self.newcanvas1.graylist))
        self.zot1.setText('XY')
        self.newcanvas1.update_data.connect(self.updateSlices1)
        self.newcanvas1.gray_data.connect(self.updateGray1)
        self.newcanvas1.new_page.connect(self.newSliceview1)
        self.slt2.setText('S %s' % (self.newcanvas2.ind + 1) + '/ %s' % (self.newcanvas2.slices))
        self.grt2.setText('G %s' % (self.newcanvas2.graylist))
        self.zot2.setText('YZ')
        self.newcanvas2.update_data.connect(self.updateSlices2)
        self.newcanvas2.gray_data.connect(self.updateGray2)
        self.newcanvas2.new_page.connect(self.newSliceview2)
        self.slt3.setText('S %s' % (self.newcanvas3.ind + 1) + '/ %s' % (self.newcanvas3.slices))
        self.grt3.setText('G %s' % (self.newcanvas3.graylist))
        self.zot3.setText('XZ')
        self.newcanvas3.update_data.connect(self.updateSlices3)
        self.newcanvas3.gray_data.connect(self.updateGray3)
        self.newcanvas3.new_page.connect(self.newSliceview3)
        if self.islinked == True:
            self.Viewpanel1.zoom_link.disconnect()
            self.Viewpanel1.move_link.disconnect()
            self.Viewpanel2.zoom_link.disconnect()
            self.Viewpanel2.move_link.disconnect()
            self.Viewpanel3.zoom_link.disconnect()
            self.Viewpanel3.move_link.disconnect()
            self.skiplink = False
            self.skipdis = True
            self.in_link.emit()

    def newSliceview1(self):
        self.grt1.setText('G %s' % (self.newcanvas1.graylist))

    def newSliceview2(self):
        self.grt2.setText('G %s' % (self.newcanvas2.graylist))

    def newSliceview3(self):
        self.grt3.setText('G %s' % (self.newcanvas3.graylist))

    def updateSlices1(self, elist):
        self.slt1.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))
        indlist.append(elist[0])
    def updateZoom1(self, data):
        self.zot1.setText('XY %s' % (data))

    def updateGray1(self, elist):
        self.grt1.setText('G %s' % (elist))

    def updateSlices2(self, elist):
        self.slt2.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))
        ind2list.append(elist[0])

    def updateZoom2(self, data):
        self.zot2.setText('YZ %s' % (data))

    def updateGray2(self, elist):
        self.grt2.setText('G %s' % (elist))

    def updateSlices3(self, elist):
        self.slt3.setText('S %s' % (elist[0] + 1) + '/ %s' % (elist[1]))
        ind3list.append(elist[0])

    def updateZoom3(self, data):
        self.zot3.setText('ZX %s' % (data))

    def updateGray3(self, elist):
        self.grt3.setText('G %s' % (elist))

    def clearWidgets(self):
        for i in reversed(range(self.count())):
            widget = self.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def setGrey1(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.newcanvas1:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas1.set_greyscale(greylist)

    def setGrey2(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.newcanvas2:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas2.set_greyscale(greylist)

    def setGrey3(self):
        maxv, minv, ok = grey_window.getData()
        if ok and self.newcanvas3:
            greylist=[]
            greylist.append(minv)
            greylist.append(maxv)
            self.newcanvas3.set_greyscale(greylist)

###################
sys._excepthook = sys.excepthook
def my_exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)

sys.excepthook = my_exception_hook

# class ExceptionHandler(QtCore.QObject):
#     errorSignal = QtCore.pyqtSignal()
#
#     def __init__(self):
#         super(ExceptionHandler, self).__init__()
#
#     def handler(self, exctype, value, traceback):
#         self.errorSignal.emit()
#         sys._excepthook(exctype, value, traceback)
# exceptionHandler = ExceptionHandler()
# sys._excepthook = sys.excepthook
# sys.excepthook = exceptionHandler.handler
#
# def something():
#     QtWidgets.QMessageBox.information('Warning', 'File error!')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyApp()
    mainWindow.showMaximized()
    mainWindow.show()
    # exceptionHandler.errorSignal.connect(something)
    sys.exit(app.exec_())