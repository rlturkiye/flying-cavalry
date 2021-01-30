# Makefile for Writing Make Files Example
 
# *****************************************************
# Variables to control Makefile operation
 
CC = g++
CFLAGS = -Wall -g
 
# ****************************************************
# Targets needed to bring the executable up to date

output: main.o message.o
	g++ main.o message.o -o output
	
main.o: main.cpp
	g++ -c main.cpp

message.o: message.cpp message.h
	g++ -c message.cpp

clean: 
	rm *.o output