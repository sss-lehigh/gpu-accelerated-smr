#!/bin/bash

cd /mu/conan/exports/compiler-options && conan create . dory/stable -pr=/mu/conan/profiles/gcc-release.profile
cd /mu/extern/ && conan create . -pr=/mu/conan/profiles/gcc-release.profile
cd /mu/shared && conan create . -pr=/mu/conan/profiles/gcc-release.profile --build=missing
cd /mu/ctrl && conan create . -pr=/mu/conan/profiles/gcc-release.profile -o log_level=OFF
cd /mu/memstore && conan create . -pr=/mu/conan/profiles/gcc-release.profile
cd /mu/conn && conan create . -pr=/mu/conan/profiles/gcc-release.profile
/mu/build.py -b release -c gcc crash-consensus
/mu/crash-consensus/libgen/export.sh gcc-release
/mu/crash-consensus/demo/using_conan_fully/build.sh gcc-release