# Projet : 

Ce projet a été réalisé dans le cadre de l'UE IAR (Intelligence Artificielle pour la Robotique) dispensée à Sorbonne Université, Paris VI.

L'objet était premièrement de reproduire les résultats obtenus par Patryk Chrabaszcz, Ilya Loshchilov et Frank Hutter dans l'article _Back to Basics: Benchmarking Canonical Evolution Strategies for Playing Atari_ avec un budget plus limité et des réseaux de neurones de plus petite taille que ceux utilisés par les auteurs. Il s'agissait dans un second temps d'étendre la comparaison OpenAI ES vs CanonicalES à d’autres algos ES tels que CEM ou CMA-ES.

### Prérequis

Pour faire tourner ce projet vous aurez besoin de :

- Python 3.5
- tensorflow (version 1.14.0)
- natsort
- python3-mpi4py
- gym[atari]

### Installations

```
sudo python3.5 get-pip.py
```
```
python3.5 -m pip install tensorflow==1.14.0
```
```
sudo python3.5 -m pip install natsort
```
```
sudo apt-get install python3-mpi4py
```
```
sudo python3.5 -m pip install gym[atari]
```

### Lancer les tests

L'ensemble du code se trouve dans le Notebook intitulé projet.ipynb.
