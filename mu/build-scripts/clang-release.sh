#!/bin/bash

export CC=clang-6.0
export CXX=clang++-6.0
export LDFLAGS=-fuse-ld=lld

cd /mu/conan/exports/compiler-options && conan create . dory/stable -pr=/mu/conan/profiles/clang-release.profile
cd /mu/extern/ && conan create . -pr=/mu/conan/profiles/clang-release.profile
cd /mu/shared && conan create . -pr=/mu/conan/profiles/clang-release.profile --build=missing
cd /mu/ctrl && conan create . -pr=/mu/conan/profiles/clang-release.profile -o log_level=OFF
cd /mu/memstore && conan create . -pr=/mu/conan/profiles/clang-release.profile
cd /mu/conn && conan create . -pr=/mu/conan/profiles/clang-release.profile
/mu/build.py -b release -c clang crash-consensus
/mu/crash-consensus/libgen/export.sh clang-release
# /mu/crash-consensus/demo/using_conan_fully/build.sh clang-release