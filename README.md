# outlier-detection-library


C'est une librarie qui comporte différents algorithmes de détection d'anomalie. Les données utilisés sont MNIST;
Les algorithmes implémentés sont classifiés en supervisés et non supervisés

Dans la famille des algo non supervisés :

Deux types d'algorithmes sont implémentés :


	- TYPE I 
	  Des algos reposant sur l'apprentissage de représentation. Ces algos consiste à apprendre une représentation des données normales et puis la représentation est utilisé 
	  pour déclencher l'alarme lors d'une anomalie arrive
		Algos implémentés : IsoForest, OneclassSVM , MixtureGaussian, Encoder (un CNN pré-entrainé utilisé comme extracteur des caractéristiques). Encodeur est souvent utilisés pour la ddétection d'anomalies conceptuelle et texturelles.	
        
        
	- TYPE II 
	  Des algos à base de l'estimation de score d'anomalie de reconstruction. Ces algos se contentent pas uniquement d'apprendre une représentation des données d'entrées, mais 
	  ils continuent à faire la reconstruction. L'anomalie est répérée lors que l'erreur de reconstruction est très importantes.
		Algos implémentés : PCA,Autoencoder(convolutionnel)

Pour tous les 2 types d'algo, le nombre de composants de la représentation des données est hyper important : si ce nombre est trop petit, alors aucune donnée ne sera reconstruite correctement,
si ce nombre est trop grand, alors les anomalies seront également reconstruites.

		
Dans la famille des algo  supervisés :  to be updated
