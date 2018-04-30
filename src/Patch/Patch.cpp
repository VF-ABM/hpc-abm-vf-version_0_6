/* 
 * File:  Patch.cpp
 *
 * File Contents: Contains Patch class
 *
 * Author: Yvonna
 * Contributors: Caroline Shung
 *               Nuttiiya Seekhao
 *               Kimberley Trickey
 *
 * Created on June 06, 2013, 4:38 AM
 *
 *****************************************************************************
 ***  Copyright (c) 2013 by A. Najafi-Yazdi                                ***
 *** This computer program is the property of Alireza Najafi-Yazd          ***
 *** and may contain confidential trade secrets.                           ***
 *** Use, examination, copying, transfer and disclosure to others,         ***
 *** in whole or in part, are prohibited except with the express prior     ***
 *** written consent of Alireza Najafi-Yazdi.                              ***
 *****************************************************************************/  // TODO(Kim): Update the file comment once we figure out the copyright issues

#include "Patch.h"
#include <iostream>
#include <atomic> 
#include "../enums.h"

using namespace std;

float Patch::numOfEachTypes[5] = {0,0,0,0,0};

Patch::Patch() { 
	this->dirty = true;
	this->occupied[read_t] = new atomic_flag(ATOMIC_FLAG_INIT);
	this->occupied[write_t] = new atomic_flag(ATOMIC_FLAG_INIT);
	this->indice[0] = 0;
	this->indice[1] = 0;
	this->indice[2] = 0;
	this->index = 0;
	this->inDamzone = false;
	this->initHA = false;
	this->initelastin = false;
	this->initcollagen = false;
	this->type[read_t] = nothing;
	this->color[read_t] = cnothing;
	this->health[read_t] = 100;
	this->damage[read_t] = 0;
	this->occupied[read_t]->clear();
	this->occupied_bool[read_t] = false;
	this->occupiedby[read_t] = unoccupied;
	this->type[write_t] = nothing;
	this->color[write_t] = cnothing;
	this->health[write_t] = 100;
	this->damage[write_t] = 0;
	this->occupied[write_t]->clear();
	this->occupied_bool[write_t] = false;
	this->occupiedby[write_t] = unoccupied;
}

Patch::Patch(int x, int y, int z, int index) {
	this->dirty = true;
	this->occupied[read_t] = new atomic_flag(ATOMIC_FLAG_INIT);
	this->occupied[write_t] = new atomic_flag(ATOMIC_FLAG_INIT);
	this->indice[0] = x;
	this->indice[1] = y;
	this->indice[2] = z;
	this->index = index;
	this->inDamzone = false;
	this->initHA = false;
	this->initelastin = false;
	this->initcollagen = false;
	this->type[read_t] = nothing;
	this->color[read_t] = cnothing;
	this->health[read_t] = 100;
	this->damage[read_t] = 0;
	this->occupied[read_t]->clear();
	this->occupied_bool[read_t] = false;
	this->occupiedby[read_t] = unoccupied;
	this->type[write_t] = nothing;
	this->color[write_t] = cnothing;
	this->health[write_t] = 100;
	this->damage[write_t] = 0;
	this->occupied[write_t]->clear();
	this->occupied_bool[write_t] = false;
	this->occupiedby[write_t] = unoccupied;
}

Patch::Patch(const Patch& obj){

  indice[0] = obj.indice[0];
  indice[1] = obj.indice[1];
  indice[2] = obj.indice[2];
  index = obj.index; 
  inDamzone = obj.inDamzone;
  initHA = obj.initHA;
  initcollagen = obj.initcollagen;
  initelastin = obj.initelastin;
  type[read_t] = obj.type[read_t];
  type[write_t] = obj.type[write_t];
  color[read_t] = obj.color[read_t];
  color[write_t] = obj.color[write_t];
  health[read_t] = obj.health[read_t];
  health[write_t] = obj.health[write_t];
  damage[read_t] = obj.damage[read_t];
  damage[write_t] = obj.damage[write_t];
  occupiedby[read_t] = obj.occupiedby[read_t];
  occupiedby[write_t] = obj.occupiedby[write_t];
  dirty = obj.dirty; 
  occupied_bool[read_t] = obj.occupied_bool[read_t];
  occupied_bool[write_t] = obj.occupied_bool[write_t];

  occupied[read_t]->clear();
  if (obj.occupied_bool[read_t]) occupied[read_t]->test_and_set(); 
  occupied[write_t]->clear();
  if (obj.occupied_bool[write_t]) occupied[read_t]->test_and_set(); 

}

Patch::~Patch() {

  delete this->occupied[read_t];
  delete this->occupied[write_t]; 

}

Patch& Patch::operator=(const Patch& obj){

  indice[0] = obj.indice[0];
  indice[1] = obj.indice[1];
  indice[2] = obj.indice[2];
  index = obj.index; 
  inDamzone = obj.inDamzone;
  initHA = obj.initHA;
  initcollagen = obj.initcollagen;
  initelastin = obj.initelastin;
  type[read_t] = obj.type[read_t];
  type[write_t] = obj.type[write_t];
  color[read_t] = obj.color[read_t];
  color[write_t] = obj.color[write_t];
  health[read_t] = obj.health[read_t];
  health[write_t] = obj.health[write_t];
  damage[read_t] = obj.damage[read_t];
  damage[write_t] = obj.damage[write_t];
  occupiedby[read_t] = obj.occupiedby[read_t];
  occupiedby[write_t] = obj.occupiedby[write_t];
  dirty = obj.dirty; 
  occupied_bool[read_t] = obj.occupied_bool[read_t];
  occupied_bool[write_t] = obj.occupied_bool[write_t];

  occupied[read_t]->clear();
  if (obj.occupied_bool[read_t]) occupied[read_t]->test_and_set(); 
  occupied[write_t]->clear();
  if (obj.occupied_bool[write_t]) occupied[read_t]->test_and_set(); 

  return *this; 
  
}

bool Patch::isOccupied() {
	return this->occupied_bool[read_t];
}

bool Patch::isOccupiedWrite() {
	return this->occupied_bool[write_t];
}

void Patch::setOccupiedLight() {
	this->dirty = true;
	this->occupied_bool[write_t] = true;
}

void Patch::setOccupiedReadLight() {
	this->occupied_bool[read_t] = true;
}

bool Patch::setOccupied() {
	this->dirty = true;
	this->occupied_bool[write_t] = true;
	return this->occupied[write_t]->test_and_set();
}

bool Patch::setOccupiedRead() {
	this->occupied_bool[read_t] = true;
	return this->occupied[read_t]->test_and_set();
}

void Patch::clearOccupied() {
	this->dirty =  true;
/* ALLOW_PATCHRACE should never be enabled except during performance debugging
 * phase to see the effects of not using atomic patch occupancy updates on the
 * performance. */
#ifdef ALLOW_PATCHRACE
	this->occupied_bool[write_t] = false;
#else
	this->occupied_bool[write_t] = false;
	this->occupied[write_t]->clear();
#endif
}

void Patch::clearOccupiedRead() {
/* ALLOW_PATCHRACE should never be enabled except during performance debugging
 * phase to see the effects of not using atomic patch occupancy updates on the
 * performance. */
#ifdef ALLOW_PATCHRACE
	this->occupied_bool[read_t] = false;
#else
	this->occupied_bool[read_t] = false;
	this->occupied[read_t]->clear();
#endif
}

int Patch::getType()
{
  return this->type[read_t];
}

bool Patch::isDamaged()
{
  return (this->damage[read_t] > 0);
}

bool Patch::isInDamZone()
{
  return this->inDamzone;
}

void Patch::lookUpType() {
	if (this->type[read_t] == tissue) {
		cout << "Patch " << this->indice[0] << ", " << this->indice[1] << ", " << this->indice[2] << " has type tissue" << endl;
	} else if (this->type[read_t] == epithelium) {
		cout << "Patch " << this->indice[0] << ", " << this->indice[1] << ", " << this->indice[2] << " has type capillary" << endl;
	} else if (this->type[read_t] == capillary) {
		cout << "Patch " << this->indice[0] << ", " << this->indice[1] << ", " << this->indice[2] << " has type epithelium" << endl;
	} else if (this->damage[read_t] != 0) {
		cout << "Patch " << this->indice[0] << ", " << this->indice[1] << ", " << this->indice[2] << " has type damage" << endl;
	} else {
		cout << "Patch " << this->indice[0] << ", " << this->indice[1] << ", " << this->indice[2] << " has unidentified type" << endl;
	}
}

int Patch::getColorfromType() {
	if (this->type[read_t] == nothing) {
		return cnothing; 
	} else if (this->type[read_t] == tissue) {
#ifdef MODEL_VOCALFOLD
                if (this->LP == SLP) return cSLP; 
                else if (this ->LP == ILP) return cILP; 
                else if (this ->LP == DLP) return cDLP; 
                else return cmuscle; 
#else
		return ctissue;  
#endif
	} else if (this->type[read_t] == epithelium) {
		return cepithelium;
	} else if (this->type[read_t] == capillary) {
		return ccapillary; 
	} else if (this->damage[read_t] != 0) {
		return cdamage;
	} else {
		cerr << "patch type is invalid!" << endl;
	}
}

bool Patch::agentOnPatch(int agentType) {
	if (this->isOccupied() == true && this->occupiedby[read_t] == agentType) return true;
	return false;
}

void Patch::updatePatch() {

	// Only update if this patch has been modified
	if (this->dirty) {
		bool occupied_next = this->occupied_bool[write_t];
		// Only update occupancy atomic flag if the occupancy has been modified
		if (occupied_next != this->occupied_bool[read_t]) {
			// If patch becomes occupied in next tick then perform set function
			if (occupied_next) this->occupied[read_t]->test_and_set();
			// If patch becomes de-occupied in next tick then perform clear function
			else this->occupied[read_t]->clear();
		}

    // Update patch class attributes
		this->type[read_t] = this->type[write_t];
		this->health[read_t] = this->health[write_t];
		this->damage[read_t] = this->damage[write_t];
		this->occupied_bool[read_t] = this->occupied_bool[write_t];
		this->occupiedby[read_t] = this->occupiedby[write_t];

    /* Prepare color attribute for visualization with Paraview 3.0 
     * (Kitware(Clifton Park, New York), Sandia National Labs(Livermore, CA),
     * CSimSoft(American Fork, Utah)). */
#ifdef PARAVIEW_RENDERING
		this->render();
#endif
		this->color[read_t] = this->color[write_t];
		this->dirty = false;
	}
}

void Patch::render() {
	if (this->isOccupied()) {
		if (this->occupiedby[read_t] == platelet) {
			this->color[write_t] = cplatelet;
		}
		if (this->occupiedby[read_t] == fibroblast) {
			this->color[write_t] = cfibroblast;
		}
		if (this->occupiedby[read_t] == neutrophil) {
			this->color[write_t] = cneutrophil;
		}
		if (this->occupiedby[read_t] == macrophag) {
			this->color[write_t] = cmacrophage;
		}
		if (this->occupiedby[read_t] == afibroblast) {
			this->color[write_t] = cafibroblast;
		}
		if (this->occupiedby[read_t]  == amacrophag) {
			this->color[write_t] = camacrophage;
		}
		if (this->occupiedby[read_t]  == aneutrophil) {
			this->color[write_t] = caneutrophil;
		}
	} else if (this->damage[read_t] > 0) {
			this->color[write_t] = 0;
	} else {
		this->color[write_t] = this->getColorfromType();
	}
}
