# CODEV - Assistant de cadrage fonctionnel pilote par IA

Application web locale qui aide a challenger une evolution ou une correction avant developpement.
CODEV simule un atelier de cadrage avec plusieurs profils client, detecte les zones floues du besoin, puis produit un score de maturite et un rapport de cadrage exploitable par une equipe projet.

L'objectif n'est pas de remplacer le chef de projet ou le developpeur, mais de faire ressortir plus tot les ambiguites qui generent souvent des bugs, des retours client ou des changements de perimetre.

## Demonstration

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
- Aide au developpeur pour preparer une reponse sans repondre a sa place.
- Indexation optionnelle de documentation projet PDF ou Markdown pour contextualiser les questions.
- Score de maturite du besoin sur plusieurs axes: completude, securite, performance, UX, donnees et exploitabilite.
- Rapport de cadrage telechargeable en HTML, imprimable en PDF depuis le navigateur.
- Analyse du rapport pour expliquer les actions qui feraient progresser le score de maturite.

Le fournisseur LLM utilise est GitHub Copilot via GitHub Models.
L'interface demande un fine-grained token GitHub ayant `Models` en lecture.

La documentation projet peut etre fournie sous forme de PDF, de fichiers Markdown, ou d'un dossier wiki Markdown.
Elle est indexee une fois dans une session documentaire locale, puis seuls les extraits utiles sont injectes dans les prompts.
Le bouton `Aide-moi a repondre` utilise la discussion et cette session documentaire pour proposer une approche de reponse au developpeur, sans repondre a sa place.
Le bouton `Generer le rapport` synthetise la discussion, calcule un score de maturite, liste les points critiques restants et propose des criteres d'acceptation.
Une fois le rapport genere, le bouton `Ameliorer le rapport` analyse les axes faibles et propose les questions, decisions et actions qui permettraient d'augmenter le score.

## Prompts utilises

Les prompts sont disponibles dans le dossier `prompts/` pour faciliter la revue ou leur modification:

- `prompts/client_questions.txt`: questions posees par le profil client selectionne.
- `prompts/opening_question.txt`: consigne utilisee pour demarrer la discussion.
- `prompts/answer_help.txt`: aide au developpeur pour preparer sa reponse.
- `prompts/framing_report.txt`: generation du rapport et du score de maturite.
- `prompts/report_improvement.txt`: analyse du rapport et actions pour ameliorer le score.

La session documentaire utilise une recherche hybride:

- embeddings `openai/text-embedding-3-small` via GitHub Models
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
$env:GITHUB_MODELS_LIGHT_MODEL="openai/gpt-4.1-nano"
$env:GITHUB_MODELS_MEDIUM_MODEL="openai/gpt-4.1-mini"
$env:GITHUB_MODELS_STRONG_MODEL="openai/gpt-4.1"
$env:GITHUB_MODELS_EMBEDDING_MODEL="openai/text-embedding-3-small"
$env:ELEVENLABS_TTS_MODEL="eleven_multilingual_v2"
```

Les tokens API sont saisis dans l'interface avant de demarrer la discussion.
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
