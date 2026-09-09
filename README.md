# CODEV - Assistant de cadrage fonctionnel pilote par IA

Application web locale qui aide a challenger une evolution ou une correction avant developpement.
CODEV simule un atelier de cadrage avec plusieurs profils client, detecte les zones floues du besoin, puis produit un score de maturite et un rapport de cadrage exploitable par une equipe projet.

L'objectif n'est pas de remplacer le chef de projet ou le developpeur, mais de faire ressortir plus tot les ambiguites qui generent souvent des bugs, des retours client ou des changements de perimetre.

## Demonstration
Cliquez pour voir la vidéo.
[<img src="https://img.youtube.com/vi/lZ-jVU_h800/hqdefault.jpg" width="600" height="300"
/>](https://www.youtube.com/embed/lZ-jVU_h800)




Le profil de l'interlocuteur client est selectionnable avant le demarrage:

- Commercial
- Technique
- Responsable

Exemple:

> Ajouter un bouton d'export Excel dans l'ecran des commandes.

CODEV peut alors faire emerger des questions comme:

- Qui est autorise a exporter les donnees ?
- Les remises negociees doivent-elles apparaitre dans l'export ?
- Quel volume de commandes doit etre supporte ?
- Quels criteres d'acceptation valident que l'export est conforme ?

## Fonctionnalites principales

- Simulation d'un atelier de cadrage avec un profil commercial, technique ou responsable.
- Questions successives pour challenger les impacts fonctionnels, les risques, les cas limites et les criteres d'acceptation.
- Detection et affichage progressifs des decisions explicitement validees, distinguees selon leur origine: documentation projet, utilisateur CODEV ou persona client (commercial, developpeur, responsable produit).
- Infobulle de tracabilite sur chaque decision avec l'extrait exact du document ou de l'echange qui justifie sa validation.
- Sauvegarde locale et reprise depuis une liste deroulante, avec la discussion, ses decisions, son cout cumule et son dernier rapport de maturite.
- Estimation en bas d'interface du cout OpenAI cumule pour la session et des tokens consommes.
- Aide au developpeur pour preparer une reponse sans repondre a sa place.
- Indexation optionnelle de documentation projet PDF ou Markdown pour contextualiser les questions.
- Score de maturite du besoin sur plusieurs axes: completude, securite, performance, UX, donnees et exploitabilite.
- Evaluation conservatrice: un axe non discute reste faible et le score global est calcule a partir des trois axes les moins matures.
- Rapport de cadrage telechargeable en HTML, imprimable en PDF depuis le navigateur.
- Analyse du rapport pour expliquer les actions qui feraient progresser le score de maturite.

Le fournisseur LLM utilise est l'API OpenAI. La cle est lue cote serveur depuis
la variable d'environnement `OPENAI_API_KEY` et n'est jamais demandee dans l'interface.

La documentation projet peut etre fournie sous forme de PDF, de fichiers Markdown, ou d'un dossier wiki Markdown.
Elle est indexee une fois dans une session documentaire locale, puis seuls les extraits utiles sont injectes dans les prompts.
Le bouton `Aide-moi a repondre` utilise la discussion et cette session documentaire pour proposer une approche de reponse au developpeur, sans repondre a sa place.
Le bouton `Generer le rapport` synthetise la discussion, calcule un score de maturite, liste les points critiques restants et propose des criteres d'acceptation.
Une fois le rapport genere, le bouton `Ameliorer le rapport` analyse les axes faibles et propose les questions, decisions et actions qui permettraient d'augmenter le score.
Le bouton `Sauvegarder` cree dans `data/sessions/` un dossier lisible compose d'un titre metier et d'un horodatage. Le titre est genere par le modele leger lors de la premiere sauvegarde, puis conserve pour toutes les sauvegardes suivantes de la meme discussion; seul l'horodatage change. Les discussions disponibles sont proposees directement dans la liste deroulante de l'interface.

La documentation projet, ses fragments et ses embeddings sont sauvegardes avec la discussion puis recharges en memoire sans nouvel appel d'embedding tant que le modele n'a pas change. Si le modele change, l'index est reconstruit et sauvegarde automatiquement. Le PDF optionnel de description de l'evolution doit en revanche etre reselectionne si la description texte ne suffit pas. Le repertoire `data/sessions/` peut contenir des documents sensibles et reste exclu de Git.

## Prompts utilises

Les prompts sont disponibles dans le dossier `prompts/` pour faciliter la revue ou leur modification:

- `prompts/client_questions.txt`: questions posees par le profil client selectionne.
- `prompts/role_commercial.txt`: posture du profil commercial.
- `prompts/role_developpeur.txt`: posture du profil technique/developpeur.
- `prompts/role_responsable_produit.txt`: posture du profil responsable produit.
- `prompts/opening_question.txt`: consigne utilisee pour demarrer la discussion.
- `prompts/answer_help.txt`: aide au developpeur pour preparer sa reponse.
- `prompts/framing_report.txt`: generation du rapport et du score de maturite.
- `prompts/report_improvement.txt`: analyse du rapport et actions pour ameliorer le score.

La session documentaire utilise une recherche hybride:

- embeddings `text-embedding-3-small` via OpenAI
- score lexical local pour conserver les correspondances exactes sur les noms d'ecrans, champs, erreurs et acronymes
- index vectoriel Python integre, accelere automatiquement par `turbovec`


## Micro

Le bouton micro utilise la reconnaissance vocale native du navigateur.
Il fonctionne surtout sur Chrome ou Edge, avec autorisation d'acces au bon micro.
L'audio n'est pas envoye au serveur: seul le texte reconnu est ajoute au champ reponse.
La touche `Ctrl` peut etre maintenue pour dicter une reponse: le relachement de `Ctrl` arrete l'ecoute.
La reponse n'est jamais envoyee automatiquement apres dictee. Elle doit etre envoyee avec le bouton `Envoyer la reponse` ou avec `Entree`.
`Maj+Entree` permet de conserver un retour a la ligne dans le champ de reponse.

## Lecture vocale

La lecture des questions du client utilise la synthese vocale native du navigateur.
Elle tourne localement avec les voix installees ou exposees par le navigateur.
Le panneau de gauche permet d'activer/desactiver la lecture, de choisir une voix et d'arreter la lecture en cours.
La sortie vocale peut aussi utiliser ElevenLabs:

- selectionner `ElevenLabs` dans `Sortie vocale`;
- saisir une cle API ElevenLabs;
- la liste des voix disponibles est chargee depuis l'API ElevenLabs avec cette cle;
- la lecture utilise ensuite la voix selectionnee.

La sortie `Locale` conserve le fonctionnement TTS du navigateur sans appel externe.


## Installation

Sous Windows, lancer:

```cmd
install.bat
```

Ou manuellement:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

```powershell
$env:OPENAI_API_KEY="votre_cle_openai"
$env:OPENAI_LIGHT_MODEL="gpt-5-nano"
$env:OPENAI_MEDIUM_MODEL="gpt-5.6-luna"
$env:OPENAI_STRONG_MODEL="gpt-4.1"
$env:OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
$env:ELEVENLABS_TTS_MODEL="eleven_multilingual_v2"
```

Seule `OPENAI_API_KEY` est obligatoire. Si elle est definie dans les variables
utilisateur ou systeme Windows, ouvrez un nouveau terminal avant de lancer l'application.
La puissance d'analyse est selectionnable dans l'interface:

- `Leger`: reponses plus rapides pour les echanges simples.
- `Moyen`: niveau par defaut pour le cadrage courant.
- `Fort`: analyse plus approfondie pour les rapports et sujets sensibles.

Les variables de modele sont optionnelles.

## Lancement

Sous Windows:

```cmd
run.bat
```

Ou manuellement:

```powershell
uvicorn app:app --reload
```

Ouvrir ensuite `http://127.0.0.1:8000`.
