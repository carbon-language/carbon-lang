# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

all: site

# Indirection for the dot-dir.
site: .gen-site

.gen-site:
	bundle exec jekyll build

run:
	bundle exec jekyll server --watch

clean:
	rm -rf .gen-site
