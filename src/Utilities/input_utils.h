/*
 * input_utils.h
 *
 * File Contents: Contains functions to manage user input for model properties
 *
 * Created on: Jun 3, 2015
 * Author: NungnunG
 * Contributors: Caroline Shung
 *               Kimberley Trickey
 */

#ifndef INPUT_UTILS_H_
#define INPUT_UTILS_H_

#pragma once

#include "../common.h"
#include <string>

using namespace std;
namespace util {

/*************************************************************************
 * MODEL OPTIONS                                                         *
 *************************************************************************/
int seed;		// Desired seed for random number genereator
int numTicks;           // Desired number of ticks for the simulation (1 tick = 30 min)
float patchWidth;       // Desired width of each patch in millimeters
float worldXwidth;      // Desired x-dimension width of the world in millimeters
float worldYwidth;      // Desired y-dimension width of the world in millimeters
float worldZwidth;      // Desired z-dimension width of the world in millimeters
char inputFileName[200]; /* Path to input file containing these inputs:
                          * Number_of_baseline_chemicals_to_be_inputed, 
                          * Baseline_TNF, Baseline_TGF, Baseline_FGF,
                          * Baseline_MMP8, Baseline_IL1beta, Baseline_IL6,
                          * Baseline_IL8, Baseline_IL10, Wound_Half_Length
                          * Wound_Depth, Wound_Severity, 
                          * Number_of_cell_to_be_inputed, 
                          * Fibroblast (initial cell count),
                          * Macrophage (initial cell count), 
                          * Neutrophil (initial cell count), Treatment_option */

/*
 * Description:	Function for getting the random seed that the user inputted
 *
 * Return: The seed that the user inputted
 *
 * Parameters: void
 */
int getSeed() {
	return seed;
}

/*
 * Description:	Function for getting the number of ticks that the user inputted
 *
 * Return: The number of ticks that the user inputted
 *
 * Parameters: void
 */
int getNumTicks() {
	return numTicks;
}

/*
 * Description:	Function for getting the patch width that the user inputted
 *
 * Return: The patch width that the user inputted
 *
 * Parameters: void
 */
float getPatchWidth() {
	return patchWidth;
}

/*
 * Description:	Function for getting the world x-width that the user inputted
 *
 * Return: The width of the world in the x dimension that the user inputted
 *
 * Parameters: void
 */
float getWorldXWidth() {
	return worldXwidth;
}

/*
 * Description:	Function for getting the world y-width that the user inputted
 *
 * Return: The width of the world in the y dimension that the user inputted
 *
 * Parameters: void
 */
float getWorldYWidth(){
	return worldYwidth;
}

/*
 * Description:	Function for getting the world z-width that the user inputted
 *
 * Return: The width of the world in the z dimension that the user inputted
 *
 * Parameters: void
 */
float getWorldZWidth() {
	return worldZwidth;
}

/*
 * Description:	Function for getting the file name containing more model inputs
 *
 * Return: The file name specified by the user that contains more model inputs
 *
 * Parameters: void
 */
char* getInputFileName() {
	return inputFileName;
}

/*
 * Description:	Function for turning command line arguments into model options
 *
 * Return: void
 *
 * Parameters: argc  -- Command line arguments count
 *             argv  -- Command line arguments
 */
void processOptions(int argc, char** argv) {

	/* Setting default options */
	seed        = 2000;
	numTicks    = 240;
#ifdef RAT_VF
        patchWidth  = 0.007;  // (mm)
#else
	patchWidth  = 0.015;  // (mm)
#endif
	worldXwidth = LPx;      //1.6;  // (mm)
	worldYwidth = LPy;      //24.9;  // (mm)
#ifdef MODEL_3D
	worldZwidth = LPz;      //17.4;  // (mm)
#else
        worldZwidth = patchWidth; 
#endif
#ifdef MODEL_VOCALFOLD
	/*
	* config_VocalFold.txt contains initial conditions (baseline chem, initial agent population). 
	* Config cell counts correspond to cellularity of entire vocal fold (1 g ww).
	* NOTE: Model initial cell counts scale with volume fraction of world to entire vocal fold (24.9mm x 1.6mm x 17.4mm)
	* [Ref] Catten, Michael, et al. "Analysis of cellular location and concentration in vocal fold lamina propria.
	* " Otolaryngology--Head and Neck Surgery 118.5 (1998): 663-667. 
	*/
	strcpy(inputFileName, "configFiles/config_VocalFold.txt");
#else
	strcpy(inputFileName, "configFiles/config.txt");
#endif

	if (argc == 1) {
		return;
	}

	// Get options from command line arguments
	for (int i = 1; i < argc; i++) {

		char* option_string = argv[i];
		if (!strcmp(option_string, "--numticks")) {
			numTicks = atoi(argv[++i]);
//		} else if (!strcmp(option_string, "--patchwidth")) {    // patch width fixed at 0.015mm 
                                                                        // maintain 1 cell max per patch occupancy
//			patchWidth = atof(argv[++i]);
		} else if (!strcmp(option_string, "--seed")) {
			seed = atoi(argv[++i]);
		} else if (!strcmp(option_string, "--wxw")) {
			worldXwidth = atof(argv[++i]);
		} else if (!strcmp(option_string, "--wyw")) {
			worldYwidth = atof(argv[++i]);
		} else if (!strcmp(option_string, "--wzw")) {
#ifdef MODEL_3D
			worldZwidth = atof(argv[++i]);
#ifdef PDE_DIFFUSE
#ifndef GPU_DIFFUSE
                        if (worldZwidth < 0.06){        // minimum z dimension for 3D PDE diffuseChem 
                                cerr << " Error: 3D wzw must be greater than 0.06mm" << endl; 
                                exit(1); 
                        }
#endif
#endif

#else
			cerr << "Error: 3D functionalities undefined. Enter 2D World dimensions. " << endl;                        
                        exit(1); 
#endif
		} else if (!strcmp(option_string, "--inputfile")) {
			strcpy(inputFileName, argv[++i]);
		} else if (!strcmp(option_string, "--help")){
			cout << "Options: " << endl;
			cout << "   --seed:          Seed for random number generator" << endl;
			cout << "   --numticks:      Number of ticks" << endl;
//			cout << "   --patchwidth:    Patch width    (mm)" << endl;
			cout << "   --wxw:           World width    (mm)" << endl;
			cout << "   --wyw:           World length   (mm)" << endl;
			cout << "   --wzw:           World height   (mm)" << endl;
			cout << "   --inputfile:     path/name of input file" << endl;
			cout << "Usage: " << endl;
			cout << "   For a 24.9mm x 17.4mm, with patch width 15 um," << endl;
			cout << "      running for 240 ticks, and input parameters from /<path_to_file>/config_large.txt:" << endl;
			cout << "      ./bin/testRun --wxw 17.4 --wyw 24.9 --numticks 240 --inputfile /<path_to_file>/config_large.txt>"
					<< endl;
			exit(-1);
		} else {
			cerr << "Error: Invalid option: " << option_string << endl;
			exit(-1);
		}
	}
}

/*
 * Description: Function for displaying model options
 *
 * Return: void
 *
 * Parameters: void
 */
void printOptions() {
	cout << "Wound Healing World ABMs Parameters:" << endl;
	cout << "	seed:		"	<< seed << endl;
	cout << "	numTicks:	"	<< numTicks << endl;
	cout << "	patchWidth:	"	<< patchWidth << endl;
	cout << "	worldXwidth:	"	<< worldXwidth << endl;
	cout << "	worldYwidth:	"	<< worldYwidth << endl;
	cout << "	worldZwidth:	"	<< worldZwidth << endl;
	cout << "	inputFileName:	"	<< inputFileName << endl;
}

}  // namespace util

#endif /* INPUT_UTILS_H_ */
