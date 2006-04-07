Summary: Static and JIT research compiler infrastructure
Name: llvm
Version: 1.6cvs
Release: 0
License: U of Illinois/NCSA Open Source License
Group: Development/Languages
Source0: llvm.tar.gz
URL: http://llvm.cs.uiuc.edu/releases/index.html
#BuildRequires: llvm-gcc
# (someday...)
BuildRoot: %{_tmppath}/%{name}-root
Requires: /sbin/ldconfig

%description
LLVM is a compiler infrastructure designed for compile-time, link-time, runtime,
and "idle-time" optimization of programs from arbitrary programming languages.
LLVM is written in C++ and has been developed since 2000 at the University of
Illinois. It currently supports compilation of C and C++ programs, using
front-ends derived from GCC 3.4. The compiler infrastructure includes mirror
sets of programming tools as well as libraries with equivalent
functionality.

%prep
%setup -q -n llvm

%build
./configure \
--prefix=%{_prefix} \
--bindir=%{_bindir} \
--datadir=%{_datadir} \
--includedir=%{_includedir} \
--libdir=%{_libdir} \
--enable-optimized \
--enable-assertions \
make

%install
rm -rf %{buildroot}
make install DESTDIR=%{buildroot}

%clean
rm -rf %{buildroot}

%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig

%files
%defattr(-, root, root)
%doc CREDITS.TXT LICENSE.TXT README.txt docs/*.{html,css,gif,jpg} docs/CommandGuide
%{_bindir}/*
%{_libdir}/*.o
%{_libdir}/*.a
%{_libdir}/*.so
%{_includedir}/llvm

%changelog
* Fri Apr 07 2006 Reid Spencer
- Make the build be optimized+assertions
* Fri May 13 2005 Reid Spencer
- Minor adjustments for the 1.5 release
* Mon Feb 09 2003 Brian R. Gaeke
- Initial working version of RPM spec file.

