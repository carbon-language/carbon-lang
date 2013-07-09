//===-- sanitizer_platform_limits_posix.h ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of Sanitizer common code.
//
// Sizes and layouts of platform-specific POSIX data structures.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_PLATFORM_LIMITS_POSIX_H
#define SANITIZER_PLATFORM_LIMITS_POSIX_H

#include "sanitizer_platform.h"

namespace __sanitizer {
  extern unsigned struct_utsname_sz;
  extern unsigned struct_stat_sz;
  extern unsigned struct_stat64_sz;
  extern unsigned struct_rusage_sz;
  extern unsigned struct_tm_sz;
  extern unsigned struct_passwd_sz;
  extern unsigned struct_group_sz;
  extern unsigned struct_sigaction_sz;
  extern unsigned siginfo_t_sz;
  extern unsigned struct_itimerval_sz;
  extern unsigned pthread_t_sz;
  extern unsigned pid_t_sz;
  extern unsigned timeval_sz;
  extern unsigned uid_t_sz;

#if !SANITIZER_ANDROID
  extern unsigned ucontext_t_sz;
#endif // !SANITIZER_ANDROID

#if SANITIZER_LINUX
  extern unsigned struct_rlimit_sz;
  extern unsigned struct_statfs_sz;
  extern unsigned struct_epoll_event_sz;
  extern unsigned struct_sysinfo_sz;
  extern unsigned struct_timespec_sz;
#endif // SANITIZER_LINUX

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  extern unsigned struct_rlimit64_sz;
  extern unsigned struct_statfs64_sz;
#endif // SANITIZER_LINUX && !SANITIZER_ANDROID

  struct __sanitizer_iovec {
    void  *iov_base;
    uptr iov_len;
  };

#if SANITIZER_MAC
  typedef unsigned long __sanitizer_pthread_key_t;
#else
  typedef unsigned __sanitizer_pthread_key_t;
#endif

#if SANITIZER_ANDROID || SANITIZER_MAC
  struct __sanitizer_msghdr {
    void *msg_name;
    unsigned msg_namelen;
    struct __sanitizer_iovec *msg_iov;
    unsigned msg_iovlen;
    void *msg_control;
    unsigned msg_controllen;
    int msg_flags;
  };
  struct __sanitizer_cmsghdr {
    unsigned cmsg_len;
    int cmsg_level;
    int cmsg_type;
  };
#else
  struct __sanitizer_msghdr {
    void *msg_name;
    unsigned msg_namelen;
    struct __sanitizer_iovec *msg_iov;
    uptr msg_iovlen;
    void *msg_control;
    uptr msg_controllen;
    int msg_flags;
  };
  struct __sanitizer_cmsghdr {
    uptr cmsg_len;
    int cmsg_level;
    int cmsg_type;
  };
#endif

#if SANITIZER_MAC
  struct __sanitizer_dirent {
    unsigned d_ino;
    unsigned short d_reclen;
    // more fields that we don't care about
  };
#elif SANITIZER_ANDROID
  struct __sanitizer_dirent {
    unsigned long long d_ino;
    unsigned long long d_off;
    unsigned short d_reclen;
    // more fields that we don't care about
  };
#else
  struct __sanitizer_dirent {
    uptr d_ino;
    uptr d_off;
    unsigned short d_reclen;
    // more fields that we don't care about
  };
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  struct __sanitizer_dirent64 {
    uptr d_ino;
    uptr d_off;
    unsigned short d_reclen;
    // more fields that we don't care about
  };
#endif

  // This thing depends on the platform. We are only interested in the upper
  // limit. Verified with a compiler assert in .cc.
  const int pthread_attr_t_max_sz = 128;
  union __sanitizer_pthread_attr_t {
    char size[pthread_attr_t_max_sz]; // NOLINT
    void *align;
  };

  uptr __sanitizer_get_sigaction_sa_sigaction(void *act);
  void __sanitizer_set_sigaction_sa_sigaction(void *act, uptr cb);
  bool __sanitizer_get_sigaction_sa_siginfo(void *act);

  const unsigned struct_sigaction_max_sz = 256;
  union __sanitizer_sigaction {
    char size[struct_sigaction_max_sz]; // NOLINT
  };

  extern uptr sig_ign;
  extern uptr sig_dfl;

#if SANITIZER_LINUX
  extern int e_tabsz;
#endif

  extern int af_inet;
  extern int af_inet6;
  uptr __sanitizer_in_addr_sz(int af);

#if SANITIZER_LINUX
  struct __sanitizer_dl_phdr_info {
    uptr dlpi_addr;
    const char *dlpi_name;
    const void *dlpi_phdr;
    short dlpi_phnum;
  };
#endif

  struct __sanitizer_addrinfo {
    int ai_flags;
    int ai_family;
    int ai_socktype;
    int ai_protocol;
#if SANITIZER_ANDROID || SANITIZER_MAC
    unsigned ai_addrlen;
    char *ai_canonname;
    void *ai_addr;
#else // LINUX
    unsigned ai_addrlen;
    void *ai_addr;
    char *ai_canonname;
#endif
    struct __sanitizer_addrinfo *ai_next;
  };

  struct __sanitizer_hostent {
    char *h_name;
    char **h_aliases;
    int h_addrtype;
    int h_length;
    char **h_addr_list;
  };

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  struct __sanitizer_glob_t {
    uptr gl_pathc;
    char **gl_pathv;
    uptr gl_offs;
    int gl_flags;
    
    void (*gl_closedir)(void *);
    void *(*gl_readdir)(void *);
    void *(*gl_opendir)(const char *);
    int (*gl_lstat)(const char *, void *);
    int (*gl_stat)(const char *, void *);
  };

  extern int glob_nomatch;
  extern int glob_altdirfunc;
#endif

  extern unsigned path_max;

#if SANITIZER_LINUX && !SANITIZER_ANDROID && \
      (defined(__i386) || defined (__x86_64))
  extern unsigned struct_user_regs_struct_sz;
  extern unsigned struct_user_fpregs_struct_sz;
  extern unsigned struct_user_fpxregs_struct_sz;

  extern int ptrace_getregs;
  extern int ptrace_setregs;
  extern int ptrace_getfpregs;
  extern int ptrace_setfpregs;
  extern int ptrace_getfpxregs;
  extern int ptrace_setfpxregs;
  extern int ptrace_getsiginfo;
  extern int ptrace_setsiginfo;
  extern int ptrace_getregset;
  extern int ptrace_setregset;
#endif

  // ioctl arguments
  struct __sanitizer_ifconf {
    int ifc_len;
    union {
      void *ifcu_req;
    } ifc_ifcu;
#if SANITIZER_MAC
  } __attribute__((packed));
#else
  };
#endif

#define IOC_SIZE(nr) (((nr) >> 16) & 0x3fff)

  extern unsigned struct_arpreq_sz;
  extern unsigned struct_ifreq_sz;
  extern unsigned struct_termios_sz;
  extern unsigned struct_winsize_sz;

#if SANITIZER_LINUX
  extern unsigned struct_cdrom_msf_sz;
  extern unsigned struct_cdrom_multisession_sz;
  extern unsigned struct_cdrom_read_audio_sz;
  extern unsigned struct_cdrom_subchnl_sz;
  extern unsigned struct_cdrom_ti_sz;
  extern unsigned struct_cdrom_tocentry_sz;
  extern unsigned struct_cdrom_tochdr_sz;
  extern unsigned struct_cdrom_volctrl_sz;
  extern unsigned struct_copr_buffer_sz;
  extern unsigned struct_copr_debug_buf_sz;
  extern unsigned struct_copr_msg_sz;
  extern unsigned struct_ff_effect_sz;
  extern unsigned struct_floppy_drive_params_sz;
  extern unsigned struct_floppy_drive_struct_sz;
  extern unsigned struct_floppy_fdc_state_sz;
  extern unsigned struct_floppy_max_errors_sz;
  extern unsigned struct_floppy_raw_cmd_sz;
  extern unsigned struct_floppy_struct_sz;
  extern unsigned struct_floppy_write_errors_sz;
  extern unsigned struct_format_descr_sz;
  extern unsigned struct_hd_driveid_sz;
  extern unsigned struct_hd_geometry_sz;
  extern unsigned struct_input_absinfo_sz;
  extern unsigned struct_input_id_sz;
  extern unsigned struct_midi_info_sz;
  extern unsigned struct_mtget_sz;
  extern unsigned struct_mtop_sz;
  extern unsigned struct_mtpos_sz;
  extern unsigned struct_rtentry_sz;
  extern unsigned struct_sbi_instrument_sz;
  extern unsigned struct_seq_event_rec_sz;
  extern unsigned struct_synth_info_sz;
  extern unsigned struct_termio_sz;
  extern unsigned struct_vt_consize_sz;
  extern unsigned struct_vt_mode_sz;
  extern unsigned struct_vt_sizes_sz;
  extern unsigned struct_vt_stat_sz;
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  extern unsigned struct_audio_buf_info_sz;
  extern unsigned struct_ax25_parms_struct_sz;
  extern unsigned struct_cyclades_monitor_sz;
  extern unsigned struct_input_keymap_entry_sz;
  extern unsigned struct_ipx_config_data_sz;
  extern unsigned struct_kbdiacrs_sz;
  extern unsigned struct_kbentry_sz;
  extern unsigned struct_kbkeycode_sz;
  extern unsigned struct_kbsentry_sz;
  extern unsigned struct_mtconfiginfo_sz;
  extern unsigned struct_nr_parms_struct_sz;
  extern unsigned struct_ppp_stats_sz;
  extern unsigned struct_scc_modem_sz;
  extern unsigned struct_scc_stat_sz;
  extern unsigned struct_serial_multiport_struct_sz;
  extern unsigned struct_serial_struct_sz;
  extern unsigned struct_sockaddr_ax25_sz;
  extern unsigned struct_unimapdesc_sz;
  extern unsigned struct_unimapinit_sz;
#endif

#if !SANITIZER_ANDROID
  extern unsigned struct_sioc_sg_req_sz;
  extern unsigned struct_sioc_vif_req_sz;
#endif

  // ioctl request identifiers

  // A special value to mark ioctls that are not present on the target platform,
  // when it can not be determined without including any system headers.
  extern unsigned IOCTL_NOT_PRESENT;

  extern unsigned IOCTL_FIOASYNC;
  extern unsigned IOCTL_FIOCLEX;
  extern unsigned IOCTL_FIOGETOWN;
  extern unsigned IOCTL_FIONBIO;
  extern unsigned IOCTL_FIONCLEX;
  extern unsigned IOCTL_FIOSETOWN;
  extern unsigned IOCTL_SIOCADDMULTI;
  extern unsigned IOCTL_SIOCATMARK;
  extern unsigned IOCTL_SIOCDELMULTI;
  extern unsigned IOCTL_SIOCGIFADDR;
  extern unsigned IOCTL_SIOCGIFBRDADDR;
  extern unsigned IOCTL_SIOCGIFCONF;
  extern unsigned IOCTL_SIOCGIFDSTADDR;
  extern unsigned IOCTL_SIOCGIFFLAGS;
  extern unsigned IOCTL_SIOCGIFMETRIC;
  extern unsigned IOCTL_SIOCGIFMTU;
  extern unsigned IOCTL_SIOCGIFNETMASK;
  extern unsigned IOCTL_SIOCGPGRP;
  extern unsigned IOCTL_SIOCSIFADDR;
  extern unsigned IOCTL_SIOCSIFBRDADDR;
  extern unsigned IOCTL_SIOCSIFDSTADDR;
  extern unsigned IOCTL_SIOCSIFFLAGS;
  extern unsigned IOCTL_SIOCSIFMETRIC;
  extern unsigned IOCTL_SIOCSIFMTU;
  extern unsigned IOCTL_SIOCSIFNETMASK;
  extern unsigned IOCTL_SIOCSPGRP;
  extern unsigned IOCTL_TIOCCONS;
  extern unsigned IOCTL_TIOCEXCL;
  extern unsigned IOCTL_TIOCGETD;
  extern unsigned IOCTL_TIOCGPGRP;
  extern unsigned IOCTL_TIOCGWINSZ;
  extern unsigned IOCTL_TIOCMBIC;
  extern unsigned IOCTL_TIOCMBIS;
  extern unsigned IOCTL_TIOCMGET;
  extern unsigned IOCTL_TIOCMSET;
  extern unsigned IOCTL_TIOCNOTTY;
  extern unsigned IOCTL_TIOCNXCL;
  extern unsigned IOCTL_TIOCOUTQ;
  extern unsigned IOCTL_TIOCPKT;
  extern unsigned IOCTL_TIOCSCTTY;
  extern unsigned IOCTL_TIOCSETD;
  extern unsigned IOCTL_TIOCSPGRP;
  extern unsigned IOCTL_TIOCSTI;
  extern unsigned IOCTL_TIOCSWINSZ;
#if (SANITIZER_LINUX && !SANITIZER_ANDROID) || SANITIZER_MAC
  extern unsigned IOCTL_SIOCGETSGCNT;
  extern unsigned IOCTL_SIOCGETVIFCNT;
#endif
#if SANITIZER_LINUX
  extern unsigned IOCTL_EVIOCGABS;
  extern unsigned IOCTL_EVIOCGBIT;
  extern unsigned IOCTL_EVIOCGEFFECTS;
  extern unsigned IOCTL_EVIOCGID;
  extern unsigned IOCTL_EVIOCGKEY;
  extern unsigned IOCTL_EVIOCGKEYCODE;
  extern unsigned IOCTL_EVIOCGLED;
  extern unsigned IOCTL_EVIOCGNAME;
  extern unsigned IOCTL_EVIOCGPHYS;
  extern unsigned IOCTL_EVIOCGRAB;
  extern unsigned IOCTL_EVIOCGREP;
  extern unsigned IOCTL_EVIOCGSND;
  extern unsigned IOCTL_EVIOCGSW;
  extern unsigned IOCTL_EVIOCGUNIQ;
  extern unsigned IOCTL_EVIOCGVERSION;
  extern unsigned IOCTL_EVIOCRMFF;
  extern unsigned IOCTL_EVIOCSABS;
  extern unsigned IOCTL_EVIOCSFF;
  extern unsigned IOCTL_EVIOCSKEYCODE;
  extern unsigned IOCTL_EVIOCSREP;
  extern unsigned IOCTL_BLKFLSBUF;
  extern unsigned IOCTL_BLKGETSIZE;
  extern unsigned IOCTL_BLKRAGET;
  extern unsigned IOCTL_BLKRASET;
  extern unsigned IOCTL_BLKROGET;
  extern unsigned IOCTL_BLKROSET;
  extern unsigned IOCTL_BLKRRPART;
  extern unsigned IOCTL_CDROMAUDIOBUFSIZ;
  extern unsigned IOCTL_CDROMEJECT;
  extern unsigned IOCTL_CDROMEJECT_SW;
  extern unsigned IOCTL_CDROMMULTISESSION;
  extern unsigned IOCTL_CDROMPAUSE;
  extern unsigned IOCTL_CDROMPLAYMSF;
  extern unsigned IOCTL_CDROMPLAYTRKIND;
  extern unsigned IOCTL_CDROMREADAUDIO;
  extern unsigned IOCTL_CDROMREADCOOKED;
  extern unsigned IOCTL_CDROMREADMODE1;
  extern unsigned IOCTL_CDROMREADMODE2;
  extern unsigned IOCTL_CDROMREADRAW;
  extern unsigned IOCTL_CDROMREADTOCENTRY;
  extern unsigned IOCTL_CDROMREADTOCHDR;
  extern unsigned IOCTL_CDROMRESET;
  extern unsigned IOCTL_CDROMRESUME;
  extern unsigned IOCTL_CDROMSEEK;
  extern unsigned IOCTL_CDROMSTART;
  extern unsigned IOCTL_CDROMSTOP;
  extern unsigned IOCTL_CDROMSUBCHNL;
  extern unsigned IOCTL_CDROMVOLCTRL;
  extern unsigned IOCTL_CDROMVOLREAD;
  extern unsigned IOCTL_CDROM_GET_UPC;
  extern unsigned IOCTL_FDCLRPRM;
  extern unsigned IOCTL_FDDEFPRM;
  extern unsigned IOCTL_FDFLUSH;
  extern unsigned IOCTL_FDFMTBEG;
  extern unsigned IOCTL_FDFMTEND;
  extern unsigned IOCTL_FDFMTTRK;
  extern unsigned IOCTL_FDGETDRVPRM;
  extern unsigned IOCTL_FDGETDRVSTAT;
  extern unsigned IOCTL_FDGETDRVTYP;
  extern unsigned IOCTL_FDGETFDCSTAT;
  extern unsigned IOCTL_FDGETMAXERRS;
  extern unsigned IOCTL_FDGETPRM;
  extern unsigned IOCTL_FDMSGOFF;
  extern unsigned IOCTL_FDMSGON;
  extern unsigned IOCTL_FDPOLLDRVSTAT;
  extern unsigned IOCTL_FDRAWCMD;
  extern unsigned IOCTL_FDRESET;
  extern unsigned IOCTL_FDSETDRVPRM;
  extern unsigned IOCTL_FDSETEMSGTRESH;
  extern unsigned IOCTL_FDSETMAXERRS;
  extern unsigned IOCTL_FDSETPRM;
  extern unsigned IOCTL_FDTWADDLE;
  extern unsigned IOCTL_FDWERRORCLR;
  extern unsigned IOCTL_FDWERRORGET;
  extern unsigned IOCTL_HDIO_DRIVE_CMD;
  extern unsigned IOCTL_HDIO_GETGEO;
  extern unsigned IOCTL_HDIO_GET_32BIT;
  extern unsigned IOCTL_HDIO_GET_DMA;
  extern unsigned IOCTL_HDIO_GET_IDENTITY;
  extern unsigned IOCTL_HDIO_GET_KEEPSETTINGS;
  extern unsigned IOCTL_HDIO_GET_MULTCOUNT;
  extern unsigned IOCTL_HDIO_GET_NOWERR;
  extern unsigned IOCTL_HDIO_GET_UNMASKINTR;
  extern unsigned IOCTL_HDIO_SET_32BIT;
  extern unsigned IOCTL_HDIO_SET_DMA;
  extern unsigned IOCTL_HDIO_SET_KEEPSETTINGS;
  extern unsigned IOCTL_HDIO_SET_MULTCOUNT;
  extern unsigned IOCTL_HDIO_SET_NOWERR;
  extern unsigned IOCTL_HDIO_SET_UNMASKINTR;
  extern unsigned IOCTL_MTIOCGET;
  extern unsigned IOCTL_MTIOCPOS;
  extern unsigned IOCTL_MTIOCTOP;
  extern unsigned IOCTL_PPPIOCGASYNCMAP;
  extern unsigned IOCTL_PPPIOCGDEBUG;
  extern unsigned IOCTL_PPPIOCGFLAGS;
  extern unsigned IOCTL_PPPIOCGUNIT;
  extern unsigned IOCTL_PPPIOCGXASYNCMAP;
  extern unsigned IOCTL_PPPIOCSASYNCMAP;
  extern unsigned IOCTL_PPPIOCSDEBUG;
  extern unsigned IOCTL_PPPIOCSFLAGS;
  extern unsigned IOCTL_PPPIOCSMAXCID;
  extern unsigned IOCTL_PPPIOCSMRU;
  extern unsigned IOCTL_PPPIOCSXASYNCMAP;
  extern unsigned IOCTL_SIOCADDRT;
  extern unsigned IOCTL_SIOCDARP;
  extern unsigned IOCTL_SIOCDELRT;
  extern unsigned IOCTL_SIOCDRARP;
  extern unsigned IOCTL_SIOCGARP;
  extern unsigned IOCTL_SIOCGIFENCAP;
  extern unsigned IOCTL_SIOCGIFHWADDR;
  extern unsigned IOCTL_SIOCGIFMAP;
  extern unsigned IOCTL_SIOCGIFMEM;
  extern unsigned IOCTL_SIOCGIFNAME;
  extern unsigned IOCTL_SIOCGIFSLAVE;
  extern unsigned IOCTL_SIOCGRARP;
  extern unsigned IOCTL_SIOCGSTAMP;
  extern unsigned IOCTL_SIOCSARP;
  extern unsigned IOCTL_SIOCSIFENCAP;
  extern unsigned IOCTL_SIOCSIFHWADDR;
  extern unsigned IOCTL_SIOCSIFLINK;
  extern unsigned IOCTL_SIOCSIFMAP;
  extern unsigned IOCTL_SIOCSIFMEM;
  extern unsigned IOCTL_SIOCSIFSLAVE;
  extern unsigned IOCTL_SIOCSRARP;
  extern unsigned IOCTL_SNDCTL_COPR_HALT;
  extern unsigned IOCTL_SNDCTL_COPR_LOAD;
  extern unsigned IOCTL_SNDCTL_COPR_RCODE;
  extern unsigned IOCTL_SNDCTL_COPR_RCVMSG;
  extern unsigned IOCTL_SNDCTL_COPR_RDATA;
  extern unsigned IOCTL_SNDCTL_COPR_RESET;
  extern unsigned IOCTL_SNDCTL_COPR_RUN;
  extern unsigned IOCTL_SNDCTL_COPR_SENDMSG;
  extern unsigned IOCTL_SNDCTL_COPR_WCODE;
  extern unsigned IOCTL_SNDCTL_COPR_WDATA;
  extern unsigned IOCTL_SNDCTL_DSP_GETBLKSIZE;
  extern unsigned IOCTL_SNDCTL_DSP_GETFMTS;
  extern unsigned IOCTL_SNDCTL_DSP_NONBLOCK;
  extern unsigned IOCTL_SNDCTL_DSP_POST;
  extern unsigned IOCTL_SNDCTL_DSP_RESET;
  extern unsigned IOCTL_SNDCTL_DSP_SETFMT;
  extern unsigned IOCTL_SNDCTL_DSP_SETFRAGMENT;
  extern unsigned IOCTL_SNDCTL_DSP_SPEED;
  extern unsigned IOCTL_SNDCTL_DSP_STEREO;
  extern unsigned IOCTL_SNDCTL_DSP_SUBDIVIDE;
  extern unsigned IOCTL_SNDCTL_DSP_SYNC;
  extern unsigned IOCTL_SNDCTL_FM_4OP_ENABLE;
  extern unsigned IOCTL_SNDCTL_FM_LOAD_INSTR;
  extern unsigned IOCTL_SNDCTL_MIDI_INFO;
  extern unsigned IOCTL_SNDCTL_MIDI_PRETIME;
  extern unsigned IOCTL_SNDCTL_SEQ_CTRLRATE;
  extern unsigned IOCTL_SNDCTL_SEQ_GETINCOUNT;
  extern unsigned IOCTL_SNDCTL_SEQ_GETOUTCOUNT;
  extern unsigned IOCTL_SNDCTL_SEQ_NRMIDIS;
  extern unsigned IOCTL_SNDCTL_SEQ_NRSYNTHS;
  extern unsigned IOCTL_SNDCTL_SEQ_OUTOFBAND;
  extern unsigned IOCTL_SNDCTL_SEQ_PANIC;
  extern unsigned IOCTL_SNDCTL_SEQ_PERCMODE;
  extern unsigned IOCTL_SNDCTL_SEQ_RESET;
  extern unsigned IOCTL_SNDCTL_SEQ_RESETSAMPLES;
  extern unsigned IOCTL_SNDCTL_SEQ_SYNC;
  extern unsigned IOCTL_SNDCTL_SEQ_TESTMIDI;
  extern unsigned IOCTL_SNDCTL_SEQ_THRESHOLD;
  extern unsigned IOCTL_SNDCTL_SYNTH_INFO;
  extern unsigned IOCTL_SNDCTL_SYNTH_MEMAVL;
  extern unsigned IOCTL_SNDCTL_TMR_CONTINUE;
  extern unsigned IOCTL_SNDCTL_TMR_METRONOME;
  extern unsigned IOCTL_SNDCTL_TMR_SELECT;
  extern unsigned IOCTL_SNDCTL_TMR_SOURCE;
  extern unsigned IOCTL_SNDCTL_TMR_START;
  extern unsigned IOCTL_SNDCTL_TMR_STOP;
  extern unsigned IOCTL_SNDCTL_TMR_TEMPO;
  extern unsigned IOCTL_SNDCTL_TMR_TIMEBASE;
  extern unsigned IOCTL_SOUND_MIXER_READ_ALTPCM;
  extern unsigned IOCTL_SOUND_MIXER_READ_BASS;
  extern unsigned IOCTL_SOUND_MIXER_READ_CAPS;
  extern unsigned IOCTL_SOUND_MIXER_READ_CD;
  extern unsigned IOCTL_SOUND_MIXER_READ_DEVMASK;
  extern unsigned IOCTL_SOUND_MIXER_READ_ENHANCE;
  extern unsigned IOCTL_SOUND_MIXER_READ_IGAIN;
  extern unsigned IOCTL_SOUND_MIXER_READ_IMIX;
  extern unsigned IOCTL_SOUND_MIXER_READ_LINE1;
  extern unsigned IOCTL_SOUND_MIXER_READ_LINE2;
  extern unsigned IOCTL_SOUND_MIXER_READ_LINE3;
  extern unsigned IOCTL_SOUND_MIXER_READ_LINE;
  extern unsigned IOCTL_SOUND_MIXER_READ_LOUD;
  extern unsigned IOCTL_SOUND_MIXER_READ_MIC;
  extern unsigned IOCTL_SOUND_MIXER_READ_MUTE;
  extern unsigned IOCTL_SOUND_MIXER_READ_OGAIN;
  extern unsigned IOCTL_SOUND_MIXER_READ_PCM;
  extern unsigned IOCTL_SOUND_MIXER_READ_RECLEV;
  extern unsigned IOCTL_SOUND_MIXER_READ_RECMASK;
  extern unsigned IOCTL_SOUND_MIXER_READ_RECSRC;
  extern unsigned IOCTL_SOUND_MIXER_READ_SPEAKER;
  extern unsigned IOCTL_SOUND_MIXER_READ_STEREODEVS;
  extern unsigned IOCTL_SOUND_MIXER_READ_SYNTH;
  extern unsigned IOCTL_SOUND_MIXER_READ_TREBLE;
  extern unsigned IOCTL_SOUND_MIXER_READ_VOLUME;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_ALTPCM;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_BASS;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_CD;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_ENHANCE;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_IGAIN;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_IMIX;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_LINE1;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_LINE2;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_LINE3;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_LINE;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_LOUD;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_MIC;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_MUTE;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_OGAIN;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_PCM;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_RECLEV;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_RECSRC;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_SPEAKER;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_SYNTH;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_TREBLE;
  extern unsigned IOCTL_SOUND_MIXER_WRITE_VOLUME;
  extern unsigned IOCTL_SOUND_PCM_READ_BITS;
  extern unsigned IOCTL_SOUND_PCM_READ_CHANNELS;
  extern unsigned IOCTL_SOUND_PCM_READ_FILTER;
  extern unsigned IOCTL_SOUND_PCM_READ_RATE;
  extern unsigned IOCTL_SOUND_PCM_WRITE_CHANNELS;
  extern unsigned IOCTL_SOUND_PCM_WRITE_FILTER;
  extern unsigned IOCTL_TCFLSH;
  extern unsigned IOCTL_TCGETA;
  extern unsigned IOCTL_TCGETS;
  extern unsigned IOCTL_TCSBRK;
  extern unsigned IOCTL_TCSBRKP;
  extern unsigned IOCTL_TCSETA;
  extern unsigned IOCTL_TCSETAF;
  extern unsigned IOCTL_TCSETAW;
  extern unsigned IOCTL_TCSETS;
  extern unsigned IOCTL_TCSETSF;
  extern unsigned IOCTL_TCSETSW;
  extern unsigned IOCTL_TCXONC;
  extern unsigned IOCTL_TIOCGLCKTRMIOS;
  extern unsigned IOCTL_TIOCGSOFTCAR;
  extern unsigned IOCTL_TIOCINQ;
  extern unsigned IOCTL_TIOCLINUX;
  extern unsigned IOCTL_TIOCSERCONFIG;
  extern unsigned IOCTL_TIOCSERGETLSR;
  extern unsigned IOCTL_TIOCSERGWILD;
  extern unsigned IOCTL_TIOCSERSWILD;
  extern unsigned IOCTL_TIOCSLCKTRMIOS;
  extern unsigned IOCTL_TIOCSSOFTCAR;
  extern unsigned IOCTL_VT_ACTIVATE;
  extern unsigned IOCTL_VT_DISALLOCATE;
  extern unsigned IOCTL_VT_GETMODE;
  extern unsigned IOCTL_VT_GETSTATE;
  extern unsigned IOCTL_VT_OPENQRY;
  extern unsigned IOCTL_VT_RELDISP;
  extern unsigned IOCTL_VT_RESIZE;
  extern unsigned IOCTL_VT_RESIZEX;
  extern unsigned IOCTL_VT_SENDSIG;
  extern unsigned IOCTL_VT_SETMODE;
  extern unsigned IOCTL_VT_WAITACTIVE;
#endif
#if SANITIZER_LINUX && !SANITIZER_ANDROID
  extern unsigned IOCTL_CYGETDEFTHRESH;
  extern unsigned IOCTL_CYGETDEFTIMEOUT;
  extern unsigned IOCTL_CYGETMON;
  extern unsigned IOCTL_CYGETTHRESH;
  extern unsigned IOCTL_CYGETTIMEOUT;
  extern unsigned IOCTL_CYSETDEFTHRESH;
  extern unsigned IOCTL_CYSETDEFTIMEOUT;
  extern unsigned IOCTL_CYSETTHRESH;
  extern unsigned IOCTL_CYSETTIMEOUT;
  extern unsigned IOCTL_EQL_EMANCIPATE;
  extern unsigned IOCTL_EQL_ENSLAVE;
  extern unsigned IOCTL_EQL_GETMASTRCFG;
  extern unsigned IOCTL_EQL_GETSLAVECFG;
  extern unsigned IOCTL_EQL_SETMASTRCFG;
  extern unsigned IOCTL_EQL_SETSLAVECFG;
  extern unsigned IOCTL_EVIOCGKEYCODE_V2;
  extern unsigned IOCTL_EVIOCGPROP;
  extern unsigned IOCTL_EVIOCSKEYCODE_V2;
  extern unsigned IOCTL_FS_IOC_GETFLAGS;
  extern unsigned IOCTL_FS_IOC_GETVERSION;
  extern unsigned IOCTL_FS_IOC_SETFLAGS;
  extern unsigned IOCTL_FS_IOC_SETVERSION;
  extern unsigned IOCTL_GIO_CMAP;
  extern unsigned IOCTL_GIO_FONT;
  extern unsigned IOCTL_GIO_SCRNMAP;
  extern unsigned IOCTL_GIO_UNIMAP;
  extern unsigned IOCTL_GIO_UNISCRNMAP;
  extern unsigned IOCTL_KDADDIO;
  extern unsigned IOCTL_KDDELIO;
  extern unsigned IOCTL_KDDISABIO;
  extern unsigned IOCTL_KDENABIO;
  extern unsigned IOCTL_KDGETKEYCODE;
  extern unsigned IOCTL_KDGETLED;
  extern unsigned IOCTL_KDGETMODE;
  extern unsigned IOCTL_KDGKBDIACR;
  extern unsigned IOCTL_KDGKBENT;
  extern unsigned IOCTL_KDGKBLED;
  extern unsigned IOCTL_KDGKBMETA;
  extern unsigned IOCTL_KDGKBMODE;
  extern unsigned IOCTL_KDGKBSENT;
  extern unsigned IOCTL_KDGKBTYPE;
  extern unsigned IOCTL_KDMAPDISP;
  extern unsigned IOCTL_KDMKTONE;
  extern unsigned IOCTL_KDSETKEYCODE;
  extern unsigned IOCTL_KDSETLED;
  extern unsigned IOCTL_KDSETMODE;
  extern unsigned IOCTL_KDSIGACCEPT;
  extern unsigned IOCTL_KDSKBDIACR;
  extern unsigned IOCTL_KDSKBENT;
  extern unsigned IOCTL_KDSKBLED;
  extern unsigned IOCTL_KDSKBMETA;
  extern unsigned IOCTL_KDSKBMODE;
  extern unsigned IOCTL_KDSKBSENT;
  extern unsigned IOCTL_KDUNMAPDISP;
  extern unsigned IOCTL_KIOCSOUND;
  extern unsigned IOCTL_LPABORT;
  extern unsigned IOCTL_LPABORTOPEN;
  extern unsigned IOCTL_LPCAREFUL;
  extern unsigned IOCTL_LPCHAR;
  extern unsigned IOCTL_LPGETIRQ;
  extern unsigned IOCTL_LPGETSTATUS;
  extern unsigned IOCTL_LPRESET;
  extern unsigned IOCTL_LPSETIRQ;
  extern unsigned IOCTL_LPTIME;
  extern unsigned IOCTL_LPWAIT;
  extern unsigned IOCTL_MTIOCGETCONFIG;
  extern unsigned IOCTL_MTIOCSETCONFIG;
  extern unsigned IOCTL_PIO_CMAP;
  extern unsigned IOCTL_PIO_FONT;
  extern unsigned IOCTL_PIO_SCRNMAP;
  extern unsigned IOCTL_PIO_UNIMAP;
  extern unsigned IOCTL_PIO_UNIMAPCLR;
  extern unsigned IOCTL_PIO_UNISCRNMAP;
  extern unsigned IOCTL_SCSI_IOCTL_GET_IDLUN;
  extern unsigned IOCTL_SCSI_IOCTL_PROBE_HOST;
  extern unsigned IOCTL_SCSI_IOCTL_TAGGED_DISABLE;
  extern unsigned IOCTL_SCSI_IOCTL_TAGGED_ENABLE;
  extern unsigned IOCTL_SIOCAIPXITFCRT;
  extern unsigned IOCTL_SIOCAIPXPRISLT;
  extern unsigned IOCTL_SIOCAX25ADDUID;
  extern unsigned IOCTL_SIOCAX25DELUID;
  extern unsigned IOCTL_SIOCAX25GETPARMS;
  extern unsigned IOCTL_SIOCAX25GETUID;
  extern unsigned IOCTL_SIOCAX25NOUID;
  extern unsigned IOCTL_SIOCAX25SETPARMS;
  extern unsigned IOCTL_SIOCDEVPLIP;
  extern unsigned IOCTL_SIOCIPXCFGDATA;
  extern unsigned IOCTL_SIOCNRDECOBS;
  extern unsigned IOCTL_SIOCNRGETPARMS;
  extern unsigned IOCTL_SIOCNRRTCTL;
  extern unsigned IOCTL_SIOCNRSETPARMS;
  extern unsigned IOCTL_SNDCTL_DSP_GETISPACE;
  extern unsigned IOCTL_SNDCTL_DSP_GETOSPACE;
  extern unsigned IOCTL_TIOCGSERIAL;
  extern unsigned IOCTL_TIOCSERGETMULTI;
  extern unsigned IOCTL_TIOCSERSETMULTI;
  extern unsigned IOCTL_TIOCSSERIAL;
#endif
}  // namespace __sanitizer

#endif

