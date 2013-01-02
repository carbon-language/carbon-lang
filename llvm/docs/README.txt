LLVM Documentation
==================

LLVM's documentation is written in reStructuredText, a lightweight
plaintext markup language (file extension `.rst`). While the
reStructuredText documentation should be quite readable in source form, it
is meant to be processed by the Sphinx documentation generation system to
create HTML pages which are hosted on <http://llvm.org/docs/> and updated
after every commit.

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
