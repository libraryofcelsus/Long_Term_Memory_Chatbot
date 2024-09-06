# Long Term Memory Chatbot
Version 0.01 of Long Term Memory Chatbot by [LibraryofCelsus.com](https://www.libraryofcelsus.com)  
  
[Installation Guide](#installation-guide)  
[Skip to Changelog](#changelog)  
[Discord Server](https://discord.gg/pb5zcNa7zE)

------
**Recent Changes**

• 09/06 Added Split API usage, allowing for background summary and memory loops to be ran locally to reduce OpenAi calls.

• 09/05 First Release

------

### What is this project?

This repository contains a simplified long-term memory system that reduces API calls compared to the one used in my Aetherius Project. Eventually, this system will serve as the chatbot component for an LLM-based RPG engine that I am currently working on. As the engine evolves, additional RPG-like modules will be integrated as their respective features are completed.

Currently, the chatbot draws its information from a v2 Character Card.  You can swap in other cards by placing them in the ./Characters folder.


------

## Memory Architecture

| Loop                           | Description                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| **User Input**                 | The interaction begins with the user sending a request to the chatbot.                                              |
| **Input Expansion**            | Enhances user input by incorporating conversation history to improve database search accuracy.                      |
| **Episodic Memory Search**     | Searches the Episodic Memory for relevant entries using the expanded user input.                                    |
| **Knowledge Domain Selection** | Identifies relevant Knowledge Domains for the user's query from the available domains.                              |
| **Additional Query Generation**| Creates additional search queries based on user input to ensure a comprehensive database search.                    |
| **Semantic Memory Search**     | Uses the generated queries to retrieve relevant semantic memories.                                                  |
| **Inner Monologue Generation** | Produces an inner monologue that reflects past experiences, consolidates search results, and deepens user input interpretation. |
| **Action Plan Generation**     | Develops an action plan based on memories and the inner monologue, functioning as an automatic chain-of-thought prompt strategy. |
| **Final Response Generation**  | Constructs a final response for the user, utilizing the generated monologue and action plan.                        |
| **Conversation Summary**       | Summarizes the conversation by condensing the last five entries when the conversation reaches a certain length, then re-integrates the summary. |
| **Episodic Memory Generation** | Creates an episodic memory from the summarized conversation entries.                                                |
| **Semantic Memory Extraction** | Extracts factual information and assigns it to the appropriate Knowledge Domain.                                    |


------


My Ai work is self-funded by my day job, consider supporting me if you appreciate my work.

<a href='https://ko-fi.com/libraryofcelsus' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi3.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

------

Join the Discord for help or to get more in-depth information!

Discord Server: https://discord.gg/pb5zcNa7zE

Made by: https://github.com/libraryofcelsus


------
# Future Plans:  
• Improve Internal Prompts   
• Add Global Events    
• Add Local Events  
• Add Individual Events  
• Add Status Effects   
• Improve Long Term Memory System  

# Changelog: 
**0.01** 

• First Release  


# Installation Guide

## Installer bat

Download the project zip folder by pressing the <> Code drop down menu or from the Releases Tab on the side.  

**Note: Project uses OpenAi and Qdrant by default.  Other Options are available in ./Settings.json**  

**1.** Install Python 3.10.6, Make sure you add it to PATH: **https://www.python.org/downloads/release/python-3106/**  

**2.** Run "install_requirements.bat" to install the needed dependencies.  The bat will install Git, and the needed python dependencies.    

(If you get an error when installing requirements run: **python -m pip cache purge**)  

**3.** Set up Qdrant or Marqo DB.  To change what DB is used, edit the "Vector_DB" Key in ./Settings.json.  Qdrant is the default.  

Qdrant Docs: https://qdrant.tech/documentation/guides/installation/    

Marqo Docs: https://docs.marqo.ai/2.9/  

To use a local Qdrant server, first install Docker: https://www.docker.com.   
Next type: **docker pull qdrant/qdrant:v1.9.1** in the command prompt.   
After it is finished downloading, type **docker run -p 6333:6333 qdrant/qdrant:v1.9.1**  

To use a local Marqo server, first install Docker: https://www.docker.com.  
Next type: **docker pull marqoai/marqo:latest** in the command prompt.  
After it is finished downloading, type **docker run --name marqo --gpus all -p 8882:8882 marqoai/marqo:latest**   

(If it gives an error, check the docker containers tab for a new container and press the start button.  Sometimes it fails to start.)  

See: https://docs.docker.com/desktop/backup-and-restore/ for how to make backups.   

Once the local Vector DB server is running, it should be auto detected by the scripts.   

**6.** Install your desired API and Model or enter OpenAi Key in ./Settings.json.  To change what Api is used, edit the "API" and "Model_Backend" Key in ./Settings.json.  
https://github.com/oobabooga/text-generation-webui   

**8.** Change the information inside of the ./Settings.json to your preferences.   To choose a Character card, change the "Character_Card_File_Name" Key.  

**9.** Launch a script with **run_Chatbot_NPC*.bat**    

**10.** Start chatting!   After you receive a response, entering 'Y' will write it to a dataset json.  Pressing 'N' will write the response to the conversation history. Pressing 'D' will delete the response for regeneration.  
  
To have the memory loops run in the background so you can start chatting immediately, set the "Memory_Output" key in the ./Settings.json to "False".  To show all of the internal loops, set "Debug_Output" to "True".  



-----

# Contact
Discord: libraryofcelsus      -> Old Username Style: Celsus#0262  

MEGA Chat: https://mega.nz/C!pmNmEIZQ 
