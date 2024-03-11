import sys

import numpy as np
from PyQt5 import QtWidgets # import PyQt5 before PyQtGraph
import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot
import calcule
from gg02 import *
from calcule import *
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        _1,_2,_3=self.graphe3()

        # Add Background color to white
        self.graphWidget.setBackground("w")
        #self.graphWidget.addLegend()
        self.forGraphe("SKRs vs SNR","SNR ","")


        self.plot(_1, _2, "SKR2", "g", "t", "left", "SKR (bits/pulsation)")
        self.plot(_1, _3, "SKR3", "r", "o", "left", "SKR (bits/pulsation)")






    def forGraphe(self,titre,basDuTitre,HautDuTitre):
        self.graphWidget.setTitle(titre, color="b", size="20pt")
        styles3 = {"color": "b", "font-size": "25px", 'textAlign': 'center'}
        self.graphWidget.setLabel("bottom",basDuTitre, **styles3)
        self.graphWidget.setLabel("top", HautDuTitre, **styles3)
    def plot(self, x, y, plotname, color,symbole,position,nomLabel):
        pen = pg.mkPen(color=color)
        label=pg.TextItem(plotname,color=color,border=2)
        label.setPos(0,max(y))
        arrow = pg.ArrowItem(pos=(0, max(y)), angle=180,)
        self.graphWidget.addItem(arrow)
        self.graphWidget.addItem(label)
        self.graphWidget.plot(x, y, name=plotname, pen=pen, symbol=symbole, symbolSize=10,symbolBrush=(color))
        styles = {"color": color, "font-size": "20px", 'textAlign': 'center'}
        self.graphWidget.setLabel(position, nomLabel, **styles)
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setBackground("w")
        self.graphWidget.show()
    def graphe(self):
        L = 50
        x = np.linspace(1, L, 5)
        loss = []
        T_star_m_tab = []
        x_star_m_tab = []
        R_M_star_tab1=[]
        R_theo_tab=[]

        for i in x:
            T = 10 ** (-A * i / 10)
            loss.append(T)
        for j in loss:
            w = np.sqrt(2) * erfinv(1 - e_PE)
            T_star_m = (np.sqrt(j) - w * np.sqrt(s_z / (M * eta * (mu - 1)))) ** 2

            T_star_m_tab.append(T_star_m)
        for k in T_star_m_tab:
            _, x_M_star, _ = key_rate_calculation(mu, k, eta, xi_star_m, v_el, beta)
            x_star_m_tab.append(x_M_star)
        for l in x_star_m_tab:
            R_M_star = beta * I_AB - l
            R_M_star_tab1.append(R_M_star)
        for m in R_M_star_tab1:
            _, R_theo, _, _, _ = composable_key_rate(n_bks, N, n, p, q, R_code,
                                                              m, x_M, p_EC, e_s, e_h,
                                                              e_cor, e_PE, H_K)

            R_theo_tab.append(R_theo)
        return x,loss,T_star_m_tab,x_star_m_tab,R_M_star_tab1,R_theo_tab
    def graphe1(self):
        beta=0.8
        eta=1
        snr = ((mu - 1) * eta * T_hat) / (1 + v_el + (eta * T_hat * xi_hat))
        x=np.linspace(0,snr,5)
        tab=[]
        for l in x:
            r_th = np.log2(alpha * np.sqrt((2 * (1 + l) ** beta) / (np.pi * np.e))) / q
            tab.append(r_th)
        return x,tab
    def graphe2(self):
        L = 50
        x = np.linspace(0, L,5 )
        eta=1
        v_el=0.5
        loss = []
        tab = []
        tab1 = []
        tab2 = []
        tab3 = []

        for i in x:
            T = 10 ** (-A * i / 10)
            loss.append(T)
        for j in loss:
            s_z = 1 + v_el + eta * j * xi  # Noise variance
            Xi = eta * j * xi  # Excess noise variance
            Chi = xi + (1 + v_el) / (j * eta)  # noise equivalent
            tab.append(s_z)
            tab1.append(Xi)
            tab2.append(Chi)
        for k in tab2:
            SNR = (mu - 1) / k  # Signal-to-noise ratio
            tab3.append(SNR)

        return x,loss, tab, tab1,tab2,tab3
    def graphe3(self):
        N=300000
        x=np.linspace(100000,N,5)
        Delta_AEP = 4 * np.log2(2 ** (1 + calcule.p / 2) + 1) * np.sqrt(np.log2(18 / ((calcule.p_EC ** 2) * (calcule.e_s ** 4))))
        Theta = np.log2(calcule.p_EC * (1 - ((calcule.e_s ** 2) / 3))) + 2 * np.log2(np.sqrt(2) * calcule.e_h)
        r_tilde_star = calcule.R_M_star - (Delta_AEP / np.sqrt(n)) + (Theta / n)
        r_m = calcule.H_K + calcule.R_code * q - p - calcule.x_M
        r_tilde = r_m - (Delta_AEP / np.sqrt(calcule.n)) + (Theta / calcule.n)
        tab=[]
        tab1=[]
        for i in x:
            r_final = ((calcule.n * calcule.p_EC) / i) * r_tilde
            tab.append(r_final)
        for j in x:
            r_theo = ((n * calcule.p_EC) / j) * r_tilde_star
            tab1.append(r_theo)
        return x,tab,tab1
app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
app.exec_()