.. _development_process:

Development Process Documentation
=================================

.. toctree::
   :hidden:

   MakefileGuide
   Projects
   LLVMBuild
   HowToReleaseLLVM

* :ref:`projects`

  How-to guide and templates for new projects that *use* the LLVM
  infrastructure.  The templates (directory organization, Makefiles, and test
  tree) allow the project code to be located outside (or inside) the ``llvm/``
  tree, while using LLVM header files and libraries.

* :doc:`LLVMBuild`

  Describes the LLVMBuild organization and files used by LLVM to specify
  component descriptions.

* :ref:`makefile_guide`

  Describes how the LLVM makefiles work and how to use them.

* :doc:`HowToReleaseLLVM`

  This is a guide to preparing LLVM releases. Most developers can ignore it.
