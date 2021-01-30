# Makefile for Writing Make Files Example
 
# *****************************************************
# Variables to control Makefile operation
 
CC = g++
CFLAGS = -Wall -g
 
# ****************************************************
# Targets needed to bring the executable up to date

output: main.o add.o
	g++ main.o add.o -o output
	
main.o: main.cpp
	g++ -c main.cpp

add.o: add.cpp add.h
	g++ -c add.cpp

clean: 
	rm *.o output