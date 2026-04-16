#!/bin/bash

cd /mu/conan/exports/compiler-options && conan create . dory/stable -pr=/mu/conan/profiles/gcc-debug.profile
cd /mu/extern/ && conan create . -pr=/mu/conan/profiles/gcc-debug.profile 
cd /mu/shared && conan create . -pr=/mu/conan/profiles/gcc-debug.profile --build=missing 
cd /mu/ctrl && conan create . -pr=/mu/conan/profiles/gcc-debug.profile -o log_level=OFF 
cd /mu/memstore && conan create . -pr=/mu/conan/profiles/gcc-debug.profile 
cd /mu/conn && conan create . -pr=/mu/conan/profiles/gcc-debug.profile 
/mu/build.py -b debug -c gcc crash-consensus 
/mu/crash-consensus/libgen/export.sh gcc-debug 
/mu/crash-consensus/demo/using_conan_fully/build.sh gcc-debug