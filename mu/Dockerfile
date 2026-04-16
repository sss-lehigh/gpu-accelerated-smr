FROM ubuntu:bionic

RUN apt-get update && apt-get install -y software-properties-common apt-transport-https ca-certificates wget gnupg && \
    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add - && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && apt-get install -y cmake

RUN apt-get update && apt-get install -y libibverbs-dev libmemcached-dev python3 python3-pip ninja-build clang lld clang-format vim tmux git memcached libevent-dev libhugetlbfs-dev libgtest-dev libnuma-dev numactl libgflags-dev

RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && make install

RUN pip3 install "conan<2.0" && rm -rf /root/.conan
COPY . /mu
WORKDIR /mu

# Clang debug
# CMD ["bash", "-c", "/mu/build-scripts/clang-debug.sh"]

# Clang release
CMD ["bash", "-c", "/mu/build-scripts/clang-release.sh"]

# gcc debug
# CMD ["bash", "-c", "/mu/build-scripts/gcc-debug.sh"]

# # gcc release
# CMD ["bash", "-c", "/mu/build-scripts/gcc-release.sh"]

# sudo docker run --privileged --rm -v $(pwd):/mu --name mu -it mu:latest && ./transfer.sh

# ** rebuild container after changing CMD **