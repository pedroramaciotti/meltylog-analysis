#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:57:20 2018

@author: pedro
"""

def thema_mapper(thema):
    #
    # For example, on Sep 2nd 2-17: 26% Series, 15% Tele, 13% Celebrites,
    # 9% Cinema, 8% Musique, 8% Jeux-Video (Accum 79%), 5% '', 4% null, 3% Sports, 
    # 2.5% Infos, 1% Not retrieved (94.5%)
    #
    if thema in ['Télé','Emissions']:
        return 'TV';
    if thema in ['Séries','Series','Ciné & Séries','Séries\\/Télé US','Séries / TV',
                 'Séries/Télé US','Séries \\/ TV','Série\\/Télé US','Série/Télé US']:
        return 'Series';        
    if thema in ['Célébrités','Stars & style','People','Social News','Celebs','Sociétés']:
        return 'Celebrities';
    if thema in ['Musique',]:
        return 'Music';
    if thema in ['Comics & Mangas','Mangas']:
        return 'Comic';
    if thema in ['Jeux-Vidéo','Games','VideoGames']:
        return 'VideoGames';
    if thema in ['Cinéma','Movies','Ciné']:
        return 'Movies';
    if thema in ['null','Not retrieved','folder_id\\','']:
        return 'Unclassifiable';
    if thema in ['Sports',"Sports d'aventure",'Sports motorisés','Sports aquatiques',
                 "Sports d'hiver",'Sports urbains']:
        return 'Sport';
    if thema in ['Actu','Info', 'News','Infos']:
        return 'News';
#    if thema in ['Sorties','Agenda','Événements']:
#        return 'Events';
#    if thema in ['Mode','Beauté','Sapes stylées','Sneakers Spot','Lookbook','Bling-bling','Marques']:
#        return 'Look';
#    if thema in ['Humour','Just for LOL','Humoristes','Se marrer']:
#        return 'Humor';
#    if thema in ['Food life','Food','fast Food']:
#        return 'Food';
#    if thema in ['Psycho - Sexo','Sexy Life']:
#        return 'Sex';
#    if thema in ['Bien être','Vivre','Healthy Life']:
#        return 'Wellbeing';
#    if thema in ['Campus','Student Spirit']:
#        return 'Student';
#    if thema in ['High-tech','Geek tips','Sciences']:
#        return 'Tech';
    else:
        return 'Other';
    
# Hot buzz
# 
#        
#    
#    
#    
#    
#    
#    
#    
