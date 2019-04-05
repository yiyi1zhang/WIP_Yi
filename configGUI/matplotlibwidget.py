# https://github.com/thomaskuestner/CNNArt

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QSizePolicy, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MyMplCanvas(FigureCanvas):
    # wheel_scroll_W_signal = pyqtSignal(int)
    wheel_scroll_signal = pyqtSignal(int,str)
    # wheel_scroll_3D_signal = pyqtSignal(int)
    # wheel_scroll_SS_signal = pyqtSignal(int)

    def __init__(self, parent=None, width=15, height=15):

        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        self.fig = plt.figure(figsize=(width, height),dpi=100)
        #self.openfile_name=''
        self.model = {}

        self.w_count=0
        self.f_count=0
        self.s_count=0
        self.layerWeights = {}  # {layer name: weights value}
        self.edgesInLayerName = [] #(input layer name, output layer name)
        self.allLayerNames = []
        self.axesDict = {}

        self.activations = {}
        self.weights ={}
        self.totalWeights=0
        self.totalWeightsSlices =0
        self.chosenWeightNumber =0
        self.chosenWeightSliceNumber =0
        self.indW =0

        self.subset_selection = {}
        self.subset_selection2 = {}

        self.chosenLayerName=[]

        self.ind =0
        self.indFS =0
        self.nrows = 0
        self.ncols = 0
        self.totalPatches = 0
        self.totalPatchesSlices = 0

        # subset selection parameters
        self.totalSS =0
        self.ssResult={}
        self.chosenSSNumber =0
        # self.alpha=0.19
        # self.Gamma=0.0000001

        self.oncrollStatus=''


        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def loadImage(self, model_png_dir):

        strImg = mpimg.imread(model_png_dir)
        ax=self.fig.add_subplot(111)
        ax.imshow(strImg)
        ax.set_axis_off()

    def plot_filters(self,chosenLayerName):

        filters = chosenLayerName.get_weights()[0]
        fig = plt.figure()
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        for j in range(len(filters)):
            ax = fig.add_subplot(j + 1)
            im = ax.matshow(filters[j][0], cmap="Greys")
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1, 0.07, 0.05, 0.821])
        fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()
        plt.show()

    def weights_plot_2D(self,chosenLayerName):
        self.fig.clf()
        self.chosenLayerName=chosenLayerName
        self.plot_weight_mosaic()

    def weights_plot_3D(self,w,chosenWeightNumber,totalWeights,totalWeightsSlices):
        self.weights=w
        self.chosenWeightNumber=chosenWeightNumber
        self.indW=self.chosenWeightNumber-1
        self.totalWeights=totalWeights
        self.totalWeightsSlices=totalWeightsSlices
        self.fig.clf()

        self.plot_weight_mosaic_3D(w)

    def weights_offset_opt(self,w):
        fig, ax = plt.subplots()

        image = w
        ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('dropped spines')


        ax.spines['left'].set_position(('outward', 2))
        ax.spines['bottom'].set_position(('outward', 2))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plt.show()

    def Weights_opt(self, matrix, max_weight=None, ax=None):

        ax = ax if ax is not None else plt.gca()

        if not max_weight:
            max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w))
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()

    def features_plot(self,chosenPatchNumber):


        self.ind = chosenPatchNumber-1

        if self.activations.ndim == 4:
            featMap=self.activations[self.ind]


            # Compute nrows and ncols for images
            n_mosaic = len(featMap)
            self.nrows = int(np.round(np.sqrt(n_mosaic)))
            self.ncols = int(self.nrows)
            if (self.nrows ** 2) < n_mosaic:
                self.ncols += 1

            self.fig.clear()
            self.plot_feature_mosaic(featMap, self.nrows, self.ncols)
            self.fig.suptitle("Feature Maps of Patch #{} ".format(self.ind+1))
            self.draw()
        else:
            pass

    def features_plot_3D(self,chosenPatchNumber,chosenPatchSliceNumber):
        self.ind = chosenPatchNumber - 1
        self.indFS =chosenPatchSliceNumber -1

        if self.activations.ndim == 5:
            featMap = self.activations[self.ind][self.indFS]
            featMap = np.transpose(featMap,(2,0,1))

            # Compute nrows and ncols for images
            n_mosaic = len(featMap)
            self.nrows = int(np.round(np.sqrt(n_mosaic)))
            self.ncols = int(self.nrows)
            if (self.nrows ** 2) < n_mosaic:
                self.ncols += 1

            self.fig.clear()
            self.plot_feature_mosaic_3D(featMap, self.nrows, self.ncols)
            self.fig.suptitle("#{} Feature Maps of Patch #{} ".format(self.indFS+1,self.ind + 1))
            self.draw()
        else:
            pass

    def subset_selection_plot(self, chosenSSNumber):

        self.chosenSSNumber =chosenSSNumber
        self.indSS = self.chosenSSNumber - 1
        # ss = self.subset_selection[self.indSS]
        ss = self.ssResult[self.indSS]
        ss=np.squeeze(ss,axis=0)

        self.fig.clear()
        self.plot_subset_mosaic(ss)
        self.draw()

    def plot_weight_mosaic(self,**kwargs):

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        #self.fig.suptitle("Weights of Layer '{}'".format(self.chosenLayerName))
        w = self.layerWeights[self.chosenLayerName]

        mosaic_number = w.shape[0]
        w = w[:mosaic_number, 0]
        nrows = int(np.round(np.sqrt(mosaic_number)))
        ncols = int(nrows)

        if nrows ** 2 < mosaic_number:
            ncols += 1

        imshape = w[0].shape

        for i in range(mosaic_number):

            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            ax.set_xlim(0, imshape[0] - 1)
            ax.set_ylim(0, imshape[1] - 1)
            mosaic = w[i]
            ax.imshow(mosaic, **kwargs)
            ax.set_axis_off()

            self.fig.suptitle("Weights of Layer '{}'".format(self.chosenLayerName))
            self.draw()

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def plot_weight_mosaic_3D(self,w,**kwargs):

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        mosaic_number = w.shape[0]
        w = w[:mosaic_number, 0] #(32,3,3,3)
        w=w[self.indW] #(3,3,3)

        nimgs = w.shape[0]
        nrows = int(np.round(np.sqrt(nimgs)))
        ncols = int(nrows)
        if (nrows ** 2) < nimgs:
            ncols += 1

        imshape = w[0].shape

        for i in range(nimgs):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            ax.set_xlim(0, imshape[0] - 1)
            ax.set_ylim(0, imshape[1] - 1)

            mosaic = w[i]

            ax.imshow(mosaic, **kwargs)
            ax.set_axis_off()

        self.fig.suptitle("#{} Weights of the Layer".format(self.indW+1))
        self.draw()
        self.oncrollStatus ='onscrollW'
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        # self.fig.canvas.mpl_connect('scroll_event', self.onscrollW)

    def plot_feature_mosaic(self,im, nrows, ncols, **kwargs):

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        nimgs = len(im)
        imshape = im[0].shape

        for i in range(nimgs):

            ax = self.fig.add_subplot(nrows, ncols,i+1)
            ax.set_xlim(0,imshape[0]-1)
            ax.set_ylim(0,imshape[1]-1)

            mosaic = im[i]

            ax.imshow(mosaic, **kwargs)
            ax.set_axis_off()
        self.draw()
        self.oncrollStatus='on_scroll'
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)

    def plot_feature_mosaic_3D(self,im, nrows, ncols, **kwargs):

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        nimgs = len(im)
        imshape = im[0].shape

        for i in range(nimgs):

            ax = self.fig.add_subplot(nrows, ncols,i+1)
            ax.set_xlim(0,imshape[0]-1)
            ax.set_ylim(0,imshape[1]-1)

            mosaic = im[i]

            ax.imshow(mosaic, **kwargs)
            ax.set_axis_off()
        self.draw()
        self.oncrollStatus = 'onscroll_3D'
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        # self.fig.canvas.mpl_connect('scroll_event', self.onscroll_3D)

    def plot_subset_mosaic(self,im,**kwargs):
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        # if len(im.shape) ==2:
        if im.ndim==2:
            imshape = im.shape

            ax = self.fig.add_subplot(111)
            ax.set_xlim(0, imshape[0] - 1)
            ax.set_ylim(0, imshape[1] - 1)
            ax.imshow(im, **kwargs)
            ax.set_axis_off()

        # elif len(im.shape) ==3:
        elif im.ndim == 3:
            im=np.transpose(im,(2,0,1))
            nimgs=im.shape[0]
            imshape = im[0].shape
            nrows = int(np.round(np.sqrt(nimgs)))
            ncols = int(nrows)
            if (nrows ** 2) < nimgs:
                ncols += 1

            for i in range(nimgs):

                ax = self.fig.add_subplot(nrows, ncols, i + 1)
                ax.set_xlim(0, imshape[0] - 1)
                ax.set_ylim(0, imshape[1] - 1)

                mosaic = im[i]

                ax.imshow(mosaic, **kwargs)
                ax.set_axis_off()
        else:
            print('the dimension of the subset selection is not right')

        self.oncrollStatus = 'onscrollSS'
        self.fig.suptitle("Subset Selection of Patch #{}".format(self.indSS+1))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        # self.fig.canvas.mpl_connect('scroll_event', self.onscrollSS)

    def on_click(self,event):
        """Enlarge or restore the selected axis."""
        ax = event.inaxes
        if ax is None:
            # Occurs when a region not in an axis is clicked...
            return
        if event.button is 1:
            # On left click_event, zoom the selected axes
            ax._orig_position = ax.get_position()
            ax.set_position([0.1, 0.1, 0.85, 0.85])
            for axis in event.canvas.figure.axes:
                # Hide all the other axes...
                if axis is not ax:
                    axis.set_visible(False)
        elif event.button is 3:
            # On right click_event, restore the axes
            try:
                ax.set_position(ax._orig_position)
                for axis in event.canvas.figure.axes:
                    axis.set_visible(True)
            except AttributeError:
                # If we haven't zoomed, ignore...
                pass
        else:
            # No need to re-draw the canvas if it's not a left or right click_event
            return
        event.canvas.draw()

    def onscrollW(self, event):

        if event.button == 'up':
            if self.indW == (self.totalWeights-1):
                pass
            else:
                self.indW+= 1
            # w = self.weights[self.indW]
            self.fig.clear()
            self.plot_weight_mosaic_3D(self.weights)
            self.draw()
            # self.wheel_scroll_W_signal.emit(self.indW+1)


        elif event.button == 'down':
            if self.indW -1<0:
                self.indW =0
            else:
                self.indW -= 1

            self.fig.clear()
            self.plot_weight_mosaic_3D(self.weights)
            self.draw()
            # self.wheel_scroll_W_signal.emit(self.indW+1)
        else:
            pass

    def onscroll(self, event):
        if self.oncrollStatus=='onscrollW':
            self.onscrollW(event)
            self.wheel_scroll_signal.emit(self.indW + 1,self.oncrollStatus)
        elif self.oncrollStatus=='on_scroll':

            if event.button == 'up':
                if self.ind == (self.totalPatches - 1):
                    pass
                else:
                    self.ind += 1
                featMap = self.activations[self.ind]
                self.fig.clear()
                self.plot_feature_mosaic(featMap, self.nrows, self.ncols)
                self.fig.suptitle("Feature Maps of Patch #{} ".format(self.ind + 1))
                self.draw()
                self.wheel_scroll_signal.emit(self.ind + 1,self.oncrollStatus)


            elif event.button == 'down':
                if self.ind - 1 < 0:
                    self.ind = 0
                else:
                    self.ind -= 1
                featMap = self.activations[self.ind]
                self.fig.clear()
                self.plot_feature_mosaic(featMap, self.nrows, self.ncols)
                self.fig.suptitle("Feature Maps of Patch #{}".format(self.ind + 1))
                self.draw()
                self.wheel_scroll_signal.emit(self.ind + 1,self.oncrollStatus)
            else:
                pass

        elif self.oncrollStatus=='onscroll_3D':
            self.onscroll_3D(event)
            self.wheel_scroll_signal.emit(self.ind + 1,self.oncrollStatus)
        elif self.oncrollStatus=='onscrollSS':
            self.onscrollSS(event)
            self.wheel_scroll_signal.emit(self.indSS + 1,self.oncrollStatus)
        else:
            pass

    def onscroll_3D(self, event):

        if event.button == 'up':
            if self.ind == (self.totalPatches - 1):
                pass
            else:
                self.ind += 1
            featMap = self.activations[self.ind][self.indFS]
            featMap = np.transpose(featMap, (2, 0, 1))
            self.fig.clear()
            self.plot_feature_mosaic_3D(featMap, self.nrows, self.ncols)
            self.fig.suptitle("#{} Feature Maps of Patch #{} ".format(self.indFS+1,self.ind + 1))
            self.draw()
            # self.wheel_scroll_3D_signal .emit(self.ind + 1)


        elif event.button == 'down':
            if self.ind - 1 < 0:
                self.ind = 0
            else:
                self.ind -= 1
            featMap = self.activations[self.ind][self.indFS]
            featMap = np.transpose(featMap, (2, 0, 1))
            self.fig.clear()
            self.plot_feature_mosaic_3D(featMap, self.nrows, self.ncols)
            self.fig.suptitle("#{} Feature Maps of Patch #{} ".format(self.indFS + 1, self.ind + 1))
            self.draw()
            # self.wheel_scroll_3D_signal .emit(self.ind + 1)
        else:
            pass

    def onscrollSS(self, event):
        if event.button == 'up':
            if self.indSS == (self.totalSS-1):
                pass
            else:
                self.indSS+= 1
            # ss = self.subset_selection[self.indSS]
            ss = self.ssResult[self.indSS]
            ss = np.squeeze(ss, axis=0)
            self.fig.clear()
            self.plot_subset_mosaic(ss)
            self.draw()
            # self.wheel_scroll_signal.emit(self.ind+1)
            # self.wheel_scroll_SS_signal.emit(self.indSS+1)


        elif event.button == 'down':
            if self.indSS -1<0:
                self.indSS =0
            else:
                self.indSS -= 1

            ss = self.ssResult[self.indSS]
            ss = np.squeeze(ss, axis=0)
            # ss = self.subset_selection[self.indSS]
            self.fig.clear()
            self.plot_subset_mosaic(ss)
            # self.fig.suptitle("Feature Maps of Patch #{} in Layer '{}'".format(self.ind + 1, self.chosenLayerName))
            self.draw()
            # self.wheel_scroll_SS_signal.emit(self.indSS+1)
        else:
            pass

    def getLayersWeights(self,LayerWeights):
        self.layerWeights = LayerWeights

    def getLayersFeatures(self,activations,totalPatches):
        self.activations = activations
        self.totalPatches=totalPatches

    def getLayersFeatures_3D(self,activations, totalPatches,totalPatchesSlices):
        self.activations = activations
        self.totalPatches=totalPatches
        self.totalPatchesSlices=totalPatchesSlices

    def getSubsetSelections(self,subset_selection,totalSS):
        self.subset_selection = subset_selection
        self.totalSS=totalSS


    def getSSResult(self,ssResult):
        self.ssResult=ssResult

class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.initUi()

    def initUi(self):
        self.layout = QVBoxLayout(self)
        self.mpl = MyMplCanvas(self, width=15, height=15)
        self.layout.addWidget(self.mpl)