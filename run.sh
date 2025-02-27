#!/usr/bin/env bash
make clean build

make run ARGS="-n=16" >output.txt