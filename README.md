# Projet IAR 

Ce projet a été réalisé dans le cadre de l'unité d'enseignement Intelligence Artificielle pour la Robotique dispensée à Sorbonne Université, Paris VI.

L'objet était premièrement de reproduire les résultats obtenus par Patryk Chrabaszcz, Ilya Loshchilov et Frank Hutter dans l'article _Back to Basics: Benchmarking Canonical Evolution Strategies for Playing Atari_ avec un budget plus limité et des réseaux de neurones de plus petite taille que ceux utilisés par les auteurs. Il s'agissait dans un second temps d'étendre la comparaison OpenAI ES vs CanonicalES à d’autres algos ES tels que CEM ou CMA-ES.

### Prérequis

Si vous souhaitez uniquement visualiser le comportement des intelligences artificielles entraînées sur les différents jeux Atari, vous aurez seulement besoin de :

- Python 3.5
- tensorflow (version 1.14.0) 
- natsort
- gym[atari]

En revanche, l'entraînement des intelligences artificielles nécessitant d'importantes ressources computationnelles, il vous faudra utiliser **Google Cloud Platform** (ou une plateforme similaire) et y créer une machine virtuelle si vous souhaitez réaliser vos propres entraînements. Il vous faudra également installer le module mpi4py.

### Installations

Tensorflow (version 1.14.0) :
```
sudo python3.5 -m pip install tensorflow==1.14.0
```
natsort :
```
sudo python3.5 -m pip install natsort
```
mpi4py :
```
sudo apt-get install python3-mpi4py
```
gym[atari] :
```
sudo python3.5 -m pip install gym[atari]
```

### Lancer les tests


