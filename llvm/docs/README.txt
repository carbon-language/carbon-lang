LLVM Documentation
==================

LLVM's documentation is written in reStructuredText, a lightweight
plaintext markup language (file extension `.rst`). While the
reStructuredText documentation should be quite readable in source form, it
is mostly meant to be processed by the Sphinx documentation generation
system to create HTML pages which are hosted on <http://llvm.org/docs/> and
updated after every commit. Manpage output is also supported, see below.

If you instead would like to generate and view the HTML locally, install
Sphinx <http://sphinx-doc.org/> and then do:

    cd docs/
    make -f Makefile.sphinx
    $BROWSER _build/html/index.html

The mapping between reStructuredText files and generated documentation is
`docs/Foo.rst` <-> `_build/html/Foo.html` <-> `http://llvm.org/docs/Foo.html`.

If you are interested in writing new documentation, you will want to read
`SphinxQuickstartTemplate.rst` which will get you writing documentation
very fast and includes examples of the most important reStructuredText
markup syntax.

Manpage Output
===============

Building the manpages is similar to building the HTML documentation. The
primary difference is to use the `man` makefile target, instead of the
default (which is `html`). Sphinx then produces the man pages in the
directory `_build/man/`.

    cd docs/
    make -f Makefile.sphinx man
    man -l _build/man/FileCheck.1

The correspondence between .rst files and man pages is
`docs/CommandGuide/Foo.rst` <-> `_build/man/Foo.1`.
These .rst files are also included during HTML generation so they are also
viewable online (as noted above) at e.g.
`http://llvm.org/docs/CommandGuide/Foo.html`.

Checking links
==============

The reachibility of external links in the documentation can be checked by
running:

    cd docs/
    make -f Makefile.sphinx linkcheck
