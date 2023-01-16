# Self-driving car
[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com) 

Start small (robotics activity with my 5 years-old child), grow slowly (remotly controlled by ssh), go far (self driving car on the road ?).


## 1. Pour commencer

### 1.1. Un peu de lecture (facultative)
Voici un peu de documentation pour se mettre dans le bain, si vous n'êtes pas déjà familier des différentes notions abordées dans ce repository.
1. Article de 2015 du journal Le Monde sur le raspberry : [lien](https://www.lemonde.fr/blog/binaire/2015/12/28/raspberry-pi-la-petite-histoire-dune-grande-idee/)
2. Court explicatif de ce qu'est un contrôlleur L298N : [lien](https://arduino.blaisepascal.fr/pont-en-h-l298n/)
3. Quelques concepts sur la reconnaissance d'image en Computer Vision, le coeur de notre système de navigation autonome : [lien](https://deepomatic.com/fr/quest-ce-que-la-reconnaissance-dimage)


### 1.2. Matériel nécessaire
1. Un chassis avec 2 roues motrices : [lien](https://www.amazon.fr/dp/B01LW6A2YU?psc=1&ref=ppx_yo2ov_dt_b_product_details).
2. Un controlleur L298N : [lien](https://www.amazon.fr/dp/B07YXFQ8CZ?psc=1&ref=ppx_yo2ov_dt_b_product_details). J'en ai acheté 2 par mesure de précaution, et en prévision d'un futur projet.
3. Un lot de 4 piles AA pour alimenter le L298N : [lien](https://www.amazon.fr/dp/B00HZV9TGS?ref=ppx_yo2ov_dt_b_product_details&th=1).
4. Un lot de câbles Dupont : [lien](https://www.amazon.fr/dp/B01JD5WCG2?psc=1&ref=ppx_yo2ov_dt_b_product_details). Véillez à avoir des mâle<->mâle, des femelle<->femelle et des mâle<->femelle. 
5. Un rapsberry pi 3B+ 1GB : [Lien](https://www.kubii.fr/cartes-raspberry-pi/2119-raspberry-pi-3-modele-b-1-gb-kubii-652508442174.html?src=raspberrypi). Vous pouvez opter pour un modèle plus récent si vous le souhaitez.
6. Une micro carte SD : [lien](https://www.amazon.fr/Hephinov-microSDHC-Adaptateur-Nintendo-Switch-Tablette/dp/B09B9GY753/ref=sr_1_1_sspa?__mk_fr_FR=%C3%85M%C3%85%C5%BD%C3%95%C3%91&crid=16NY2OYVRODBM&keywords=raspberry%2Bpi%2B3%2Bsd%2Bcard&qid=1673821358&sprefix=raspberry%2Bpi3%2Bsd%2Bcard%2Caps%2C78&sr=8-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1). Si vous en possedez une de 8Go, ça devrait suffire.
7. Une batterie externe pour alimenter le raspberry : [lien](https://www.amazon.fr/dp/B07R4YVBND?psc=1&ref=ppx_yo2ov_dt_b_product_details).
8. Quelque morceaux de carton. Une surface totale maximum de 30 cm par 30 cm.
9. Trois ou quatre élastiques
10. Un pistolet à colle ou de la super glue
11. Un rouleau adhésif d'isolation électrique : [lien](https://www.leroymerlin.fr/produits/electricite-domotique/rallonge-multiprise-enrouleur-et-cable-electrique/accessoires-de-connexion-boite-de-derivation/accessoires-de-electricite/ruban-adhesif-l-10-m-x-l-19-mm-noir-75361223.html)

## 2. Assemblage

### 2.1. Installation de l'OS sur le raspberry

### 2.2. Chassis

### 2.3. Premier test


## 3. Préparation de l'IA pour la reconnaissance d'image
Scénario simple : identifier un objet sur une scène
### 3.1. Collecte des données

### 3.2. Nettoyage des données

### 3.3. Annotation des données

### 3.5. Préparation du moteur d'entrainement

### 3.5. Entrainement et test


## 4. Préparation des commandes de pilotage des roues motrices par l'IA
Scénario simple : poursuite d'une cible dans le champ de vision







