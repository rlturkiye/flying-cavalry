# Project Setup

## Linux 

### 0- Create Epic Games Account
In order to access Unreal Engine Source code you have to request access it.

(If you have problems ask me - Ugurkan)

### 1- Download Unreal Engine 4.25

Notice : It will take 10GB of download. If you have slow internet do execute this command before you go sleep and don't turn off your PC.

```
git clone -b 4.25 https://github.com/EpicGames/UnrealEngine.git
cd UnrealEngine
./Setup.sh
./GenerateProjectFiles.sh
make
```

After git clone rest will also take 2 hours to build(depending on your RAM-CPU Speed)

Congrats you have now Unreal Engine 4.25 built from source code.

### 2- Install AirSim and Build It

Clone AirSim and build it:

```
# go to the folder where you clone GitHub projects
git clone https://github.com/Microsoft/AirSim.git
cd AirSim

```

```
./setup.sh
./build.sh
```

### 3- Test your Install

Go to UnrealEngine installation folder and start Unreal by running ./Engine/Binaries/Linux/UE4Editor.

When Unreal Engine prompts for opening or creating project, select Browse and choose AirSim/Unreal/Environments/Blocks (or your custom Unreal project).

Alternatively, the project file can be passed as a commandline argument. For Blocks: ./Engine/Binaries/Linux/UE4Editor <AirSim_path>/Unreal/Environments/Blocks/Blocks.uproject

If you get prompts to convert project, look for More Options or Convert-In-Place option.

 If you get prompted to build, choose Yes. If you get prompted to disable AirSim plugin, choose No.

After Unreal Editor loads, press Play button.

## Windows
BEFORE START INSTALL GIT

https://git-scm.com/download/win
64 bit Windows Setup ( not the portable one )


### 0- Create Epic Games Account 

If you don't have one already.

### 1- Install Epic Games Store

https://www.unrealengine.com/en-US/download/creators?install=true

Choose "Creators" license.


### 2- Install Unreal Engine 4.25 from Epic Games Store 

From launcher SELECT **UNREAL ENGINE 4.25**
definitely **not** select 4.26 or newer.

If you can't see it let us know !


### 3- Download Airsim

* Install Visual Studio 2019. Make sure to select Desktop Development with C++ and Windows 10 SDK 10.0.18362 (should be selected by default) while installing VS 2019.

* Start Developer Command Prompt for VS 2019.
Clone the repo: git clone https://github.com/Microsoft/AirSim.git, and go the AirSim directory by cd AirSim.

Note: It's generally not a good idea to install AirSim in C drive. This can cause scripts to fail, and requires running VS in Admin mode. Instead clone in a different drive such as D or E.

* Run build.cmd from the command line. This will create ready to use plugin bits in the Unreal\Plugins folder that can be dropped into any Unreal project.

### 4- Test Install

## Set-up Unreal Environment
* Navigate to folder AirSim\Unreal\Environments\Blocks and run update_from_git.bat .

* Double click on generated .sln file to open in Visual Studio 2019 or newer.

* Make sure Blocks project is the startup project in Unreal Engine.

* Make sure build configuration is set to DebugGame_Editor and Win64 in Visual Studio. Hit F5 to run. This should open blocks project in Unreal Engine.

* Press the Play button in Unreal Editor and you will see something like this,
Note: If Unreal Editor asks you to "Choose Vehicle" just click "No" to select drone.
!(https://github.com/rlturkiye/flying-cavalry/blob/main/images/unreal_ss.jpg?raw=true)

If you see a drone, you have finished the installation successfully!
Note: If you see a gray screen after pressing the play button, just wait a few minutes for loading the assets then you can see the environment and a drone.
