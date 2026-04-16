#!/bin/bash

export CC=clang-6.0
export CXX=clang++-6.0
export LDFLAGS=-fuse-ld=lld

rm -rf ~/.conan/data
cd /mu/conan/exports/compiler-options && conan create . dory/stable -pr=/mu/conan/profiles/clang-debug.profile
cd /mu/extern/ && conan create . -pr=/mu/conan/profiles/clang-debug.profile
cd /mu/shared && conan create . -pr=/mu/conan/profiles/clang-debug.profile --build=missing
cd /mu/ctrl && conan create . -pr=/mu/conan/profiles/clang-debug.profile -o log_level=OFF
cd /mu/memstore && conan create . -pr=/mu/conan/profiles/clang-debug.profile
cd /mu/conn && conan create . -pr=/mu/conan/profiles/clang-debug.profile
/mu/build.py -b debug -c clang crash-consensus
/mu/crash-consensus/libgen/export.sh clang-debug
/mu/crash-consensus/demo/using_conan_fully/build.sh clang-debug