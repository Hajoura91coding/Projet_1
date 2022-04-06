#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
from scipy.stats import skew
from scipy.stats import kurtosis
from datetime import datetime
import os
get_ipython().run_line_magic('matplotlib', 'inline')
# In[5]:


class P2() :

    def __init__(self):
        pass

    def load_data(self, path):
        # importation des fichiers de données
        data = pd.read_csv(path, sep=',')
        return data

    #Pie plot with matplotlib
    def camembert(self, data, columns, title_fig):
        fig = plt.figure(figsize=(16, 8), dpi= 60, facecolor='w', edgecolor='k')
        ax = plt.subplot(1,1, 1)

        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}%".format(pct, absolute)

        wedges, texts, autotexts = ax.pie(data[columns].value_counts(),
                                           autopct=lambda pct: func(pct, data[columns].value_counts())
                                           ,textprops=dict(color="w"))
        ax.legend(wedges, data[columns].value_counts().index,
                  title=columns,
                  loc="center left",
                  fontsize = 20,
                   bbox_to_anchor=(1, 0, 0.5, 1))
        ax.set_title(title_fig, fontsize = 20)
        plt.setp(autotexts, size=15, weight="bold")
        plt.show()

    #Pourcentage de valeurs manquante dans le dataframe
    def nan_percent(self, data):
        nan_percents = round(data.isna().sum().sum()/data.size * 100,0)
        return print(nan_percents,"%")

    # Fonction permettant de recupérer un indicateur pour un niveau donnée (Pays, Region, etc..)
    # Elle permet egalement de rajouter les colonnes Region, Income group au dataset de sortie
    def recup_indicateur(self, dataset, niveau,indicateur, Country, CountrySeries):
        #Liste des années de 1970 à 2100
        annees = list(map(str,range(1970,2100,1)))
        #Liste des pays présents dans le fichier CountrySeries et Country => 211 pays en tous
        PaysValide = Country.loc[Country["Country Code"].isin(CountrySeries['CountryCode'].unique().tolist()),'Short Name'].tolist()
        RegionValide = Country['Region'].dropna().unique().tolist()
        if niveau == 'Region':
            checklist = RegionValide
        else:
            checklist = PaysValide
        tab = dataset[dataset['Country Name'].isin(checklist)]
        ColonneAnnee = [yy for yy in annees if yy in dataset.columns]

        # On supprime toutes les lignes dont toutes la valeurs sont à NaN sur la base des colonnes de subset
        # On supprime les colonnes dont toutes la valeurs sont à NaN
        tab = tab.loc[tab['Indicator Name'] == indicateur].dropna(how = 'all', subset=ColonneAnnee).dropna(axis=1,how='all')
        #Rajout des colonnes utiles à l'analyse
        if (not tab.empty):
            tab = tab.drop(['Indicator Code'], axis=1)
            tab1 = Country[['Country Code', 'Region','Income Group']]
            newTab = tab1.merge(tab,on='Country Code')
        return newTab

    # avec la librairie seaborn
    def hist(self,data,label_x, label_y, title):
        plt.figure(figsize=(10,9))
        sns.set()
        sns.displot(data, kde = True)
        plt.xlabel(label_x)
        plt.ylabel(label_y, c='k')
        plt.title(title)
        plt.show()
    # préliminaire afin d'afficher le taux de scolarisation en fonction des revenues des pays
    def group_taux_scol():
        # on range les dates dans une liste de 1990 à 2016
        Col_AgrG = list(map(str,range(1990,2016,1)))
        # on ajoute a cette liste Income Group
        Col_AgrG.append('Income Group')
        # la moyenne du taux de scolarité de tous les pays selon les revenus par année
        TauxScolIncomeGroup = TauxScolarisationpParPays[Col_AgrG].groupby(['Income Group']).mean()
        # on place Income group au début du dataframe
        TauxScolIncomeGroup.reset_index(level=0, inplace=True)
        return TauxScolIncomeGroup

    def boxplot(self, data, title):
        plt.figure(figsize=(10,7))
        sns.set()
        sns.boxplot(data = data)
        plt.title(title)
        plt.show()

    def barplot(self, data, x, y, label_x, label_y, title):
        plt.figure(figsize = (10,7))
        sns.barplot(data = data, x=x, y =y)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(title)
        plt.show()

    # préliminaire afin d'afficher le taux de scolarisation en fonction des revenues des pays
    def group_taux_scol(self, TauxScolarisationpParPays):
        # on range les dates dans une liste de 1990 à 2016
        Col_AgrG = list(map(str,range(1990,2016,1)))
        # on ajoute a cette liste Income Group
        Col_AgrG.append('Income Group')
        # la moyenne du taux de scolarité de tous les pays selon les revenus par année
        TauxScolIncomeGroup = TauxScolarisationpParPays[Col_AgrG].groupby(['Income Group']).mean()
        # on place Income group au début du dataframe
        TauxScolIncomeGroup.reset_index(level=0, inplace=True)
        return TauxScolIncomeGroup

    def fig_evol_TauxScol_grRevenue(self, TauxScolIncomeGroup):
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle("Evolution du Taux de scolarisation par groupe de revenu", fontsize  = 14)
        # on convertit les dates en datetime
        Date_Deb = datetime.strptime('2000-01-01','%Y-%m-%d')
        Date_Fin= datetime.strptime('2015-12-31','%Y-%m-%d')
        # on stocke les dates dans une liste nommée années_ref
        annees_ref = pd.date_range(Date_Deb,Date_Fin,freq='Y')
        # on parcours la liste des Income group de 1990 à 2015 et on trace l'évolution du taux de scolarité
        for IG in TauxScolIncomeGroup['Income Group'].to_list():
            Scol = TauxScolIncomeGroup[TauxScolIncomeGroup['Income Group']==IG].loc[:,'2000':'2015'].values
            plt.plot(annees_ref,Scol.T,linewidth = 3.0, label=IG)

        #plt.ylim(0,100)

        plt.xlabel('Année')
        plt.ylabel('Evolution du Taux de scolarisation')

        leg = plt.legend(loc='upper center', ncol=2, mode="expand", shadow=True, fancybox=False)
        leg.get_frame().set_alpha(0.5)
        plt.show()

    def bar_pie(self, data,database, ylabel, title):
        width = 0.2
        x = np.arange(len(data.index))
        #dpi résolution de l'image
        fig = plt.figure(figsize=(16, 8), dpi= 60, facecolor='w', edgecolor='k')

        ax1 = plt.subplot(1,2, 1)

        bar1 = ax1.bar(x - width,data['MEAN'].values,width,label='Mean')
        bar2 = ax1.bar(x,data['MEDIAN'].values,width,label='Median')
        bar3 = ax1.bar(x + width,data['STD'].values,width,label='Std')

        ax1.set_ylabel(ylabel,fontsize=20)
        ax1.set_title(title,fontsize=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(data.index,rotation=90,fontsize=20)
        ax1.legend()

        def func(pct, allvals):
            absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}%".format(pct, absolute)

        colors = ['lightskyblue', 'red', 'blue', 'green', 'gold','deeppink','wheat']

        ax2 = plt.subplot(1,2, 2)

        wedges, texts, autotexts = ax2.pie(database['Region'].value_counts(),
                                           autopct=lambda pct: func(pct, database['Region'].value_counts())
                                           ,colors=colors,textprops=dict(color="w"))

        ax2.legend(wedges, database['Region'].value_counts().index,
                  loc="center",
                  fontsize = 20,
                  bbox_to_anchor=(0.25, -0.7, 0.5, 1))
        ax2.set_title('Nombre de pays par région', fontsize = 20)

        plt.setp(autotexts, size=15, weight="bold")

        plt.show()


    def map(self, data, year, title):
        colors = 9
        cmap = 'Blues'
        figsize = (16, 40)
        year = year
        title = title.format(year)
        imgfile = 'img/{}.png'.format(title)
        ax = data.dropna().plot(column=year, cmap=cmap, figsize=figsize, scheme='equal_interval', k=colors, legend=True)
        data[data.isna().any(axis=1)].plot(ax=ax)

        ax.set_title(title, fontdict={'fontsize': 20}, loc='center')
        ##ax.annotate(description, xy=(0.1, 0.1), size=12, xycoords='figure fraction')

        ax.set_axis_off()
        ax.set_xlim([-1.5e7, 1.7e7])
        ax.get_legend().set_bbox_to_anchor((.12, .4))
        ##ax.get_figure()

        ax.set_title(title, fontdict={'fontsize': 20}, loc='center')
        ##ax.annotate(description, xy=(0.1, 0.1), size=12, xycoords='figure fraction')

        ax.set_axis_off()
        ax.set_xlim([-1.5e7, 1.7e7])
        ax.get_legend().set_bbox_to_anchor((.12, .4))
        ##ax.get_figure()

    def metrique(self,GDPtri,TriPopJeune,TriPopOrdi ,Acc,NTJ):
        #Je renomme les colonnes
        GDPtri.rename(columns={'2011': 'PIB'}, inplace=True)
        TriPopJeune.rename(columns={'2015': 'Population jeune'}, inplace=True)
        TriPopOrdi.rename(columns={'2006': 'Ordinateur personnel(100 personnes)'}, inplace=True)
        Acc.rename(columns={"2015": "Accès à internet(100 personnes)"}, inplace=True)
        NTJ.rename(columns = { "Lycee 2007":"Inscrit Lycée" , "Terti 2007":"Inscrit dans le tertiaire"}, inplace=True)
        #je fusionne les datframes entre eux pour en faire un tableau
        df1 = GDPtri.merge(TriPopJeune[["Country Name", "Population jeune"]], on = "Country Name", how = "outer")
        df2 = df1.merge(TriPopOrdi[["Country Name", "Ordinateur personnel(100 personnes)"]],on = "Country Name",how = "outer")
        df3 = df2.merge(Acc[["Country Name", "Accès à internet(100 personnes)"]],on = "Country Name",how = "outer")
        df4 = df3.merge(NTJ[["Country Name", "Inscrit Lycée","Inscrit dans le tertiaire"]], on = "Country Name" ,how = "outer")
        return df1, df2, df3, df4

    def evol_pib(self,GPD):
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle("Evolution du PIB", fontsize  = 14)
        Date_Deb = datetime.strptime('2000-01-01','%Y-%m-%d')
        Date_Fin= datetime.strptime('2015-12-31','%Y-%m-%d')
        annees_ref = pd.date_range(Date_Deb,Date_Fin,freq='Y')

        Etats_unis   = GPD[GPD['Country Code'] == 'USA'].loc[:,'2000':'2015'].values
        Royaume_unis = GPD[GPD['Country Code'] == 'GBR'].loc[:,'2000':'2015'].values
        France       = GPD[GPD['Country Code'] == 'FRA'].loc[:,'2000':'2015'].values
        Espagne      = GPD[GPD['Country Code'] == 'ESP'].loc[:,'2000':'2015'].values
        Chine        = GPD[GPD['Country Code'] == 'CHN'].loc[:,'2000':'2015'].values
        Japon        = GPD[GPD['Country Code'] == 'JPN'].loc[:,'2000':'2015'].values

        plt.plot(annees_ref,Etats_unis.T,linewidth = 3.0, label='Etats_unis')
        plt.plot(annees_ref,Royaume_unis.T,linewidth = 3.0, label='Royaume_unis')

        plt.plot(annees_ref,France.T,linewidth = 3.0, label='France')
        plt.plot(annees_ref,Espagne.T,linewidth = 3.0, label='Espagne')

        plt.plot(annees_ref,Chine.T,linewidth = 3.0, label='Chine')
        plt.plot(annees_ref,Japon.T,linewidth = 3.0, label='Japon')

        plt.xlabel('Année')
        plt.ylabel('PIB')

        leg = plt.legend(loc='upper center', ncol=2, mode="expand", shadow=True, fancybox=False)
        leg.get_frame().set_alpha(0.5)
        plt.show()

    def evol_nom_ordi(self, PopulationOrdinateur):
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle("Evolution du nombre d'ordinateurs personels", fontsize  = 14)
        Date_Deb = datetime.strptime('2000-01-01','%Y-%m-%d')
        Date_Fin= datetime.strptime('2009-12-31','%Y-%m-%d')
        annees_ref = pd.date_range(Date_Deb,Date_Fin,freq='Y')

        Etats_unis   = PopulationOrdinateur[PopulationOrdinateur['Country Code'] == 'USA'].loc[:,'2000':'2009'].values
        Royaume_unis = PopulationOrdinateur[PopulationOrdinateur['Country Code'] == 'GBR'].loc[:,'2000':'2009'].values
        France       = PopulationOrdinateur[PopulationOrdinateur['Country Code'] == 'FRA'].loc[:,'2000':'2009'].values
        Espagne      = PopulationOrdinateur[PopulationOrdinateur['Country Code'] == 'ESP'].loc[:,'2000':'2009'].values
        Chine        = PopulationOrdinateur[PopulationOrdinateur['Country Code'] == 'CHN'].loc[:,'2000':'2009'].values
        Japon        = PopulationOrdinateur[PopulationOrdinateur['Country Code'] == 'JPN'].loc[:,'2000':'2009'].values

        plt.plot(annees_ref,Etats_unis.T,linewidth = 3.0, label='Etats_unis')
        plt.plot(annees_ref,Royaume_unis.T,linewidth = 3.0, label='Royaume_unis')

        plt.plot(annees_ref,France.T,linewidth = 3.0, label='France')
        plt.plot(annees_ref,Espagne.T,linewidth = 3.0, label='Espagne')

        plt.plot(annees_ref,Chine.T,linewidth = 3.0, label='Chine')
        plt.plot(annees_ref,Japon.T,linewidth = 3.0, label='Japon')

        plt.xlabel('Année')
        plt.ylabel('nombre ordinateurs personels')

        leg = plt.legend(loc='upper center', ncol=2, mode="expand", shadow=True, fancybox=False)
        leg.get_frame().set_alpha(0.5)
        plt.show()

    def acc_net(self, Accenet):
        fig = plt.figure(figsize=(10, 7))
        fig.suptitle("Evolution de l'accés à internet", fontsize  = 14)
        Date_Deb = datetime.strptime('2000-01-01','%Y-%m-%d')
        Date_Fin= datetime.strptime('2016-12-31','%Y-%m-%d')
        annees_ref = pd.date_range(Date_Deb,Date_Fin,freq='Y')

        Etats_unis   = Accenet[Accenet['Country Code'] == 'USA'].loc[:,'2000':'2016'].values
        Royaume_unis = Accenet[Accenet['Country Code'] == 'GBR'].loc[:,'2000':'2016'].values
        France       = Accenet[Accenet['Country Code'] == 'FRA'].loc[:,'2000':'2016'].values
        Espagne      = Accenet[Accenet['Country Code'] == 'ESP'].loc[:,'2000':'2016'].values
        Chine        = Accenet[Accenet['Country Code'] == 'CHN'].loc[:,'2000':'2016'].values
        Japon        = Accenet[Accenet['Country Code'] == 'JPN'].loc[:,'2000':'2016'].values

        plt.plot(annees_ref,Etats_unis.T,linewidth = 3.0, label='Etats_unis')
        plt.plot(annees_ref,Royaume_unis.T,linewidth = 3.0, label='Royaume_unis')

        plt.plot(annees_ref,France.T,linewidth = 3.0, label='France')
        plt.plot(annees_ref,Espagne.T,linewidth = 3.0, label='Espagne')

        plt.plot(annees_ref,Chine.T,linewidth = 3.0, label='Chine')
        plt.plot(annees_ref,Japon.T,linewidth = 3.0, label='Japon')

        plt.xlabel('Année')
        plt.ylabel('Accés à internet')

        leg = plt.legend(loc='upper center', ncol=2, mode="expand", shadow=True, fancybox=False)
        leg.get_frame().set_alpha(0.5)
        plt.show()
# In[ ]:
