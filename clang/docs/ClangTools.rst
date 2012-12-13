===========
Clang Tools
===========

Clang Tools are standalone command line (and potentially GUI) tools
design for use by C++ developers who are already using and enjoying
Clang as their compiler. These tools provide developer-oriented
functionality such as fast syntax checking, automatic formatting,
refactoring, etc.

Only a couple of the most basic and fundamental tools are kept in the
primary Clang Subversion project. The rest of the tools are kept in a
side-project so that developers who don't want or need to build them
don't. If you want to get access to the extra Clang Tools repository,
simply check it out into the tools tree of your Clang checkout and
follow the usual process for building and working with a combined
LLVM/Clang checkout:

-  With Subversion:

   -  ``cd llvm/tools/clang/tools``
   -  ``svn co http://llvm.org/svn/llvm-project/clang-tools-extra/trunk       extra``

-  Or with Git:

   -  ``cd llvm/tools/clang/tools``
   -  ``git clone http://llvm.org/git/clang-tools-extra.git extra``

This document describes a high-level overview of the organization of
Clang Tools within the project as well as giving an introduction to some
of the more important tools. However, it should be noted that this
document is currently focused on Clang and Clang Tool developers, not on
end users of these tools.

Clang Tools Organization
========================

Clang Tools are CLI or GUI programs that are intended to be directly
used by C++ developers. That is they are *not* primarily for use by
Clang developers, although they are hopefully useful to C++ developers
who happen to work on Clang, and we try to actively dogfood their
functionality. They are developed in three components: the underlying
infrastructure for building a standalone tool based on Clang, core
shared logic used by many different tools in the form of refactoring and
rewriting libraries, and the tools themselves.

The underlying infrastructure for Clang Tools is the
:doc:`LibTooling <LibTooling>` platform. See its documentation for much
more detailed information about how this infrastructure works. The
common refactoring and rewriting toolkit-style library is also part of
LibTooling organizationally.

A few Clang Tools are developed along side the core Clang libraries as
examples and test cases of fundamental functionality. However, most of
the tools are developed in a side repository to provide easy separation
from the core libraries. We intentionally do not support public
libraries in the side repository, as we want to carefully review and
find good APIs for libraries as they are lifted out of a few tools and
into the core Clang library set.

Regardless of which repository Clang Tools' code resides in, the
development process and practices for all Clang Tools are exactly those
of Clang itself. They are entirely within the Clang *project*,
regardless of the version control scheme.

Core Clang Tools
================

The core set of Clang tools that are within the main repository are
tools that very specifically compliment, and allow use and testing of
*Clang* specific functionality.

``clang-check``
~~~~~~~~~~~~~~~

This tool combines the LibTooling framework for running a Clang tool
with the basic Clang diagnostics by syntax checking specific files in a
fast, command line interface. It can also accept flags to re-display the
diagnostics in different formats with different flags, suitable for use
driving an IDE or editor. Furthermore, it can be used in fixit-mode to
directly apply fixit-hints offered by clang.

FIXME: Link to user-oriented clang-check documentation.

Extra Clang Tools
=================

As various categories of Clang Tools are added to the extra repository,
they'll be tracked here. The focus of this documentation is on the scope
and features of the tools for other tool developers; each tool should
provide its own user-focused documentation.
