FROM quay.io/fenicsproject/dev:latest

ENV GMSH_VERSION="gmsh-4.4.1"
ENV GMSH_FOLDER=""$GMSH_VERSION"-Linux64-sdk"
ENV GMSH_URL="http://gmsh.info/bin/Linux/"$GMSH_FOLDER".tgz"
ENV PYTHONPATH=""

USER root

RUN apt-get update && \
    apt install -y git && \
    apt install -y mercurial && \
    apt-get install -y wget && \
    apt-get install -y software-properties-common && \
    apt-get install -y libglu1 libxft2 libxcursor-dev libxinerama-dev && \
    apt-get install -y libx11-dev bison flex automake libtool libxext-dev libncurses-dev python3-dev xfonts-100dpi cython3 python3-scipy make zlib1g-dev 

    
USER fenics

# Get Gmsh
RUN mkdir gmsh && \
    cd gmsh && \
    wget $GMSH_URL && \
    ls && \
    echo ""$GMSH_VERSION"-Linux64-sdk.tgz" && \
    tar xvzf ""$GMSH_VERSION"-Linux64-sdk.tgz" && \
    rm ""$GMSH_VERSION"-Linux64-sdk.tgz"

RUN mkdir ulfy && \
    git clone https://github.com/mirok/ulfy.git  && \
    cd ulfy  && \
    git checkout python3  && \
    python3 setup.py install --user 

ENV PYTHONPATH="/home/fenics/gmsh/$GMSH_FOLDER/lib":"$PYTHONPATH"
ENV PYTHONPATH "${PYTHONPATH}:/home/fenics/ulfy"


USER root

# Release

# docker run -it -v $(pwd):/home/fenics/shared brainpulse:latest
