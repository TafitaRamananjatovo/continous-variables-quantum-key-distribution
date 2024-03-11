import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QPropertyAnimation
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QMainWindow, QApplication
import calcule
from gg02 import key_rate_calculation
import ui_main
from calcule import *
from key_generation import *
from qiskit import QuantumCircuit, execute, Aer
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = ui_main.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.ui.minimize_window_button.clicked.connect(lambda: self.showMinimized())
        self.ui.exit_button.clicked.connect(lambda: self.close())
        self.ui.restore_window_button.clicked.connect(lambda: self.restore_or_maximize_window())
        self.ui.init_btn.clicked.connect(lambda: self.plot_clear())
        self.get_widget_lineEdit()
        self.load_graphe_with_attenuation()

        #self.ui.btn_graphe.clicked.connect(lambda: self.show_graphe())
        #self.ui.draw_btn.clicked.connect(lambda :self.show_graphe())
        self.ui.menu_res_1.clicked.connect(lambda : self.ui.stackedWidget.setCurrentWidget(self.ui.page_1))
        self.ui.menu_res_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))
        self.ui.menu_res_3.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))
        self.ui.menu_icon.clicked.connect(lambda : self.slideLeftMenu())
        self.ui.res_one.clicked.connect(lambda : self.the_depends_results())
        self.ui.res_two.clicked.connect(lambda : self.final_results())

        self.ui.load_key.clicked.connect(lambda : self.about_key())
        self.ui.pushButton.clicked.connect(lambda :self.toDecimale())

    def get_widget_lineEdit(self):
        L = self.ui.longeur_2
        L.setValidator(QDoubleValidator(L))
        L.setText(str(5))

        A = self.ui.Attenuation
        A.setValidator(QDoubleValidator(A))
        A.setText(str(0.2))

        xi = self.ui.xi
        xi.setValidator(QDoubleValidator(xi))
        xi.setText(str(0.01))

        eta = self.ui.eta
        eta.setValidator(QDoubleValidator(eta))
        eta.setText(str(0.8))

        v_el = self.ui.v_el
        v_el.setValidator(QDoubleValidator(v_el))
        v_el.setText(str(0.1))

        beta = self.ui.beta
        beta.setValidator(QDoubleValidator(beta))
        beta.setText(str(0.9225))

        n_bks = self.ui.n_bks
        n_bks.setValidator(QDoubleValidator(n_bks))
        n_bks.setText(str(5))

        N = self.ui.Number_of_blocks
        N.setValidator(QDoubleValidator(N))
        N.setText(str(500000))


        # les valeurs de sorties

        T = self.ui.T
        T.setReadOnly(True)
        T.setValidator(QDoubleValidator(T))

        s_z = self.ui.s_z
        s_z.setReadOnly(True)
        s_z.setValidator(QDoubleValidator(s_z))

        Xi = self.ui.Xi
        Xi.setReadOnly(True)
        Xi.setValidator(QDoubleValidator(Xi))

        Chi = self.ui.Chi
        Chi.setReadOnly(True)
        Chi.setValidator(QDoubleValidator(Chi))

        # les resultats de simulation

        SNR = self.ui.SNR
        SNR.setReadOnly(True)
        SNR.setValidator(QDoubleValidator(SNR))

        SNR_hat = self.ui.SNR_hat
        SNR_hat.setReadOnly(True)
        SNR_hat.setValidator(QDoubleValidator(SNR_hat))

        p_EC = self.ui.p_EC
        p_EC.setReadOnly(True)
        p_EC.setValidator(QDoubleValidator(p_EC))

        FER = self.ui.FER
        FER.setReadOnly(True)
        FER.setValidator(QDoubleValidator(FER))

        I_AB = self.ui.I_AB_2
        I_AB.setReadOnly(True)
        I_AB.setValidator(QDoubleValidator(I_AB))

        I_AB_hat = self.ui.I_AB_hat
        I_AB_hat.setReadOnly(True)
        I_AB_hat.setValidator(QDoubleValidator(I_AB_hat))

        H_K = self.ui.H_K
        H_K.setReadOnly(True)
        H_K.setValidator(QDoubleValidator(H_K))

        x_M = self.ui.x_M
        x_M.setReadOnly(True)
        x_M.setValidator(QDoubleValidator(x_M))

        x_Ey = self.ui.x_Ey
        x_Ey.setReadOnly(True)
        x_Ey.setValidator(QDoubleValidator(x_Ey))


        R_code = self.ui.R_code
        R_code.setReadOnly(True)
        R_code.setValidator(QDoubleValidator(R_code))

        epsilon = self.ui.epsilon
        epsilon.setReadOnly(True)
        epsilon.setValidator(QDoubleValidator(epsilon))

        R_final = self.ui.R_final
        R_final.setReadOnly(True)
        R_final.setValidator(QDoubleValidator(R_final))

        Rasy = self.ui.R_asy
        Rasy.setReadOnly(True)
        Rasy.setValidator(QDoubleValidator(Rasy))

        Rm = self.ui.R_m
        Rm.setReadOnly(True)
        Rm.setValidator(QDoubleValidator(Rm))

        RM = self.ui.R_M
        RM.setReadOnly(True)
        RM.setValidator(QDoubleValidator(RM))

        #resultat de la generation de cle quantique
        bits_Alice=self.ui.bits_Alice
        bits_Alice.setReadOnly(True)

        bits_Bob = self.ui.bits_Bob
        bits_Bob.setReadOnly(True)

        basis_AB = self.ui.basis_AB
        basis_AB.setReadOnly(True)

        key_AB = self.ui.key_AB
        key_AB.setReadOnly(True)

        return L, A, xi, eta, v_el, beta, n_bks, N, T, s_z, Xi, Chi, SNR, \
        SNR_hat, p_EC, FER, I_AB, I_AB_hat, H_K, \
        x_M, x_Ey, Rasy, Rm, RM,R_code,epsilon,R_final,bits_Alice,bits_Bob,basis_AB,key_AB
    def toDecimale(self):
        key = self.ui.key_AB
        key.setReadOnly(True)
        key_decimal=self.about_key()
        final_res = toDecimal(key_decimal)
        key.setPlainText(str(final_res))
    def about_key(self):
        bits_alice=self.ui.bits_Alice
        bits_alice.setReadOnly(True)
        bits_bob=self.ui.bits_Bob
        bits_bob.setReadOnly(True)
        basis_bob=self.ui.basis_AB
        basis_bob.setReadOnly(True)
        key = self.ui.key_AB
        key.setReadOnly(True)


        bits_alice_res, bits_bob_res, basis_bob_res, key_res = generate_key(100)
        str_res = toString(key_res)
        str_Alice=toString(bits_alice_res)
        str_bob=toString(bits_bob_res)
        basis=toString(basis_bob_res)


        bits_alice.setPlainText(str(str_Alice))
        bits_bob.setPlainText(str(str_bob))
        basis_bob.setPlainText(str(basis))
        key.setPlainText(str(str_res))

        return str_res

    def the_depends_results(self):
        L, A, xi, eta, v_el, beta, n_bks, N, T, s_z, Xi, Chi, SNR, \
        SNR_hat, p_EC, FER, I_AB, I_AB_hat, H_K,\
        x_M, x_Ey, Rasy, Rm, RM, R_code, epsilon, R_final, bits_Alice, bits_Bob, basis_AB, key_AB = self.get_widget_lineEdit()


        T_res, s_z_res, Xi_res, Chi_res, SNR_res = calcule.depedent_values(float(calcule.mu),float(L.text()),
                                                            float(A.text()), float(xi.text()),
                                                            float(eta.text()), float(v_el.text()),
                                                            float(n_bks.text()), float(N.text()),
                                                            float(calcule.p), float(calcule.q))

        T.setText(str(round(T_res, 2)))
        s_z.setText(str(round(s_z_res, 2)))
        Xi.setText(str(round(Xi_res, 2)))
        Chi.setText(str(round(Chi_res, 2)))
        SNR.setText(str(round(SNR_res, 2)))




        # T.setText(_1)s_z.setText(_2)Xi.setText(_3)Chi.setText(_4) M.setText(_5)m.setText(_6)n.setText(_7)Gf.setText(_8)d.setText(_9)SNR.setText(_10)
    def final_results(self):
        L, A, xi, eta, v_el, beta, n_bks, N, T, s_z, Xi, Chi, SNR, \
        SNR_hat, p_EC, FER,  I_AB, I_AB_hat, H_K, \
        x_M, x_Ey, Rasy, Rm, RM, R_code, epsilon, R_final, bits_Alice, bits_Bob, basis_AB, key_AB = self.get_widget_lineEdit()


        SNR_hat_res,p_EC_res,FER_res,H_K_res,x_M_res,x_Ey_res,R_asy_res, R_M_res, R_m_res, I_AB_res,\
          I_AB_hat_res,R_code_res, epsilon_res,R_res= calcule.secure_key_rate_graphe(float(L.text()),
                                                            float(xi.text()),
                                                            float(eta.text()),
                                                            float(v_el.text()),
                                                            float(beta.text()),
                                                            int(n_bks.text()),
                                                            int(N.text()),
                                                            int(calcule.q),
                                                            float(calcule.e_PE))

        SNR_hat.setText(str(round(SNR_hat_res, 2)))
        I_AB.setText(str(round(I_AB_res, 2)))
        I_AB_hat.setText(str(round(I_AB_hat_res, 2)))
        H_K.setText(str(round(H_K_res, 2)))
        x_M.setText(str(round(x_M_res, 2)))
        x_Ey.setText(str(round(x_Ey_res, 2)))
        Rasy.setText(str(round(R_asy_res, 2)))
        RM.setText(str(round(R_M_res, 2)))
        Rm.setText(str(round(R_m_res, 2)))
        R_code.setText(str(round(R_code_res, 2)))
        epsilon.setText(str(round(epsilon_res,3)))
        R_final.setText(str(round(R_res, 2)))
        SNR_hat.setText(str(round(SNR_hat_res, 2)))
        p_EC.setText(str(round(p_EC_res, 2)))
        FER.setText(str(round(FER_res, 2)))
    def show_graphe(self):

        if self.ui.SKR_Th_radio.isChecked():
            self.plot_clear()
            self.graphe()
        elif self.ui.SNR_radio.isChecked():
            self.plot_clear()
            self.load_graphe_with_attenuation()
        elif self.ui.Code_rate_radio.isChecked():
            self.plot_clear()
            self.graphe1()
        else:
            self.plot_clear()
            self.graphe3()
    def load_graphe_with_distance(self,L):

        x = np.linspace(1, L, 5)
        loss = []
        rate_asy = []
        Inf_mutuel = []
        x_Ey_tab = []
        for i in x:
            T = 10 ** (-A * i / 10)
            loss.append(T)
        for j in loss:
            I_AB, x_Ey, R_asy = key_rate_calculation(mu, j, eta, xi, v_el, beta)
            rate_asy.append(R_asy)
            Inf_mutuel.append(I_AB)
            x_Ey_tab.append(x_Ey)

        self.plot_data_simple(x, Inf_mutuel, "Trasmittance","t","Trasmittance","right", "g","Taux en fonction de la distance"," distance de transmission (km) ","")
        self.plot_data_simple(x, rate_asy, "Taux de génération de clé asymptotique","s","Taux de génération de clé asymptotique","left","b","Taux en fonction de la distance"," distance de transmission (km) ","")
        self.plot_data_simple(x, x_Ey_tab, "Taux de génération de clé dans les mauvais cas","+","Taux de génération de clé dans les mauvais cas","right","r","Taux en fonction de la distance"," distance de transmission (km) ","")
        self.show()
    def load_graphe_with_attenuation(self):
        L = 50
        x = np.linspace(1, L, 5)
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

        self.plot_data_simple(x, tab, "Variance de bruit ", "t", "Variance de bruit ", "right", "g",
                       "Signal,Bruit et distance",
                   "distance de transmission (km)", "")
        self.plot_data_simple(x, tab2, "Bruit Equivalent", "s", "Bruit Equivalent", "right", "b",
                   "Signal,Bruit et distance", " distance de transmission (km) ", "")
        self.plot_data_simple(x, tab3, "SNR", "+", "SNR", "left", "r",
                       "Signal,Bruit et distance", " distance de transmission (km) ", "")
        self.show()

    """def graphe(self):
        L = 50
        x = np.linspace(1, L, 5)
        loss = []
        T_star_m_tab = []
        x_star_m_tab = []
        R_M_star_tab1 = []
        R_theo_tab = []

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

        self.plot_data_simple(x, R_M_star_tab1, "Trasmittance", "t", "Trasmittance", "right", "g", "Taux en fonction de la distance",
                          " distance de transmission (km) ", "")
        self.plot_data_simple(x, R_theo_tab, "Taux de génération de clé asymptotique", "s",
                          "Taux de génération de clé asymptotique", "left", "b", "Taux en fonction de la distance",
                          " distance de transmission (km) ", "")
        self.show()"""
    """def graphe1(self):
        snr = ((mu - 1) * eta * T_hat) / (1 + v_el + (eta * T_hat * xi_hat))
        x=np.linspace(0,snr,5)
        tab=[]
        for l in x:
            r_th = np.log2(alpha * np.sqrt((2 * (1 + l) ** beta) / (np.pi * np.e))) / q
            tab.append(r_th)
        self.plot_data_simple(x, tab, "Trasmittance", "t", "Trasmittance", "right", "g",
                              "Taux en fonction de la distance",
                              " distance de transmission (km) ", "")
        self.show()
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
        self.plot_data_simple(x, tab, "Trasmittance", "t", "Trasmittance", "right", "g",
                              "Taux en fonction de la distance",
                              " distance de transmission (km) ", "")
        self.plot_data_simple(x, tab1, "Taux de génération de clé asymptotique", "s",
                              "Taux de génération de clé asymptotique", "left", "b", "Taux en fonction de la distance",
                              " distance de transmission (km) ", "")
        self.show()"""

    def restore_or_maximize_window(self):
        # If window is maxmized
        if self.isMaximized():
            self.showNormal()
            # Change Icon
            self.ui.restore_window_button.setIcon(QtGui.QIcon(u":/images/maximize.svg"))
        else:
            self.showMaximized()
            # Change Icon
            self.ui.restore_window_button.setIcon(QtGui.QIcon(u":/images/minimize-2.svg"))
    def slideLeftMenu(self):
        # Get current left menu width
        width = self.ui.toggle_menu.width()
        # If minimized
        if width == 0:
            # Expand menu
            newWidth = 200
            self.ui.restore_window_button.setIcon(QtGui.QIcon(u":/images/arrow-right.svg"))
        # If maximized
        else:
            # Restore menu
            newWidth = 0
            self.ui.restore_window_button.setIcon(QtGui.QIcon(u":/images/menu.svg"))
        # Animate the transition
        self.animation = QPropertyAnimation(self.ui.toggle_menu, b"maximumWidth")  # Animate minimumWidht
        self.animation.setDuration(100)
        self.animation.setStartValue(width)  # Start value is the current menu width
        self.animation.setEndValue(newWidth)  # end value is the new menu width
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.start()

    def plot_data_simple(self, Xaxis, Yaxis,nom,symbole, nomL,position, couleur,titre,basDuTitre,HautDuTitre):
        pen = pg.mkPen(color=couleur,width=3,style=QtCore.Qt.DashLine)
        label = pg.TextItem(nom, color=couleur, border=None)
        label.setPos(max(Xaxis), max(Yaxis))
        self.ui.graphe.addItem(label)
        self.ui.graphe.plot(Xaxis, Yaxis, name=nom, pen=pen, symbol=symbole, symbolSize=5, symbolBrush=(couleur))
        self.ui.graphe.setBackground("w")
        styles = {"color": couleur, "font-size": "10px", 'textAlign': 'center'}
        self.ui.graphe.setLabel(position, nomL, **styles)
        self.ui.graphe.showGrid(x=True, y=True)
        self.ui.graphe.setTitle(titre, color="b", size="10pt")
        styles3 = {"color": "b", "font-size": "10px", 'textAlign': 'center'}
        self.ui.graphe.setLabel("bottom", basDuTitre, **styles3)
        self.ui.graphe.setLabel("top", HautDuTitre, **styles3)

    def plot_clear(self):
        self.ui.graphe.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
