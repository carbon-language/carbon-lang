# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM centos:centos7.9.2009

# install
RUN yum -y install which zlib zlib-devel curl perl-ExtUtils-CBuilder perl-ExtUtils-MakeMaker gcc gcc-c++ curl-devel expat-devel gettext-devel openssl-devel expect make wget gettext zip unzip scl-utils \
    && yum -y install centos-release-scl \
    && yum -y install devtoolset-7-gcc* \
    && mkdir -p /data/App \
    && cd /data/App \
    && wget https://github.com/git/git/archive/refs/tags/v2.37.1.tar.gz \
    && tar -zxvf v2.37.1.tar.gz \
    && wget https://curl.haxx.se/download/curl-7.70.0.tar.gz --no-check-certificate \
    && tar -zxvf curl-7.70.0.tar.gz \
    && rm *.tar.gz

RUN source /opt/rh/devtoolset-7/enable \
    && gcc --version \
    && cd /data/App/git-2.37.1 \
    && make prefix=/usr/local/git all \
    && make prefix=/usr/local/git install \
    && cd ../curl-7.70.0 \
    && ./configure --prefix=/usr/local/curl \
    && make && make install \
    && cd .. \
    && export PATH=/usr/local/git/bin:/usr/local/curl/bin:$PATH \
    && NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
    && echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /root/.bash_profile \
    && eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

RUN export PATH=/home/linuxbrew/.linuxbrew/bin/:/home/linuxbrew/.linuxbrew/opt/llvm/bin:/usr/local/git/bin:/usr/local/curl/bin:$PATH \
    && export HOMEBREW_GIT_PATH=/usr/local/git/bin/git \
    && export HOMEBREW_CURL_PATH=/usr/local/curl/bin/curl \
    && brew update \
    && brew install bazelisk \
    && brew install libxcrypt --overwrite \
    && brew install --force-bottle --only-dependencies llvm \
    && brew install --force-bottle --force --verbose llvm \
    && brew info llvm \
    && brew install node \
    && brew install python@3.9 \
    && pip3 install -U pip \
    && pip3 install pre-commit \
    && pip3 install black \
    && pip3 install codespell

ENV LANG=en_US.UTF-8 \
    HOMEBREW_GIT_PATH=/usr/local/git/bin/git \
    HOMEBREW_CURL_PATH=/usr/local/curl/bin/curl \
    PATH=/home/linuxbrew/.linuxbrew/opt/llvm/bin:/home/linuxbrew/.linuxbrew/bin/:/home/linuxbrew/.linuxbrew/Cellar/llvm/14.0.6_1.reinstall/bin:/usr/local/git/bin:/usr/local/curl/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
