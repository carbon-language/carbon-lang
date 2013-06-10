//===-- sanitizer_common_interceptors_ioctl.inc -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Ioctl handling in common sanitizer interceptors.
//===----------------------------------------------------------------------===//

#include "sanitizer_flags.h"

struct ioctl_desc {
  unsigned req;
  // FIXME: support read+write arguments. Those are currently marked as WRITE.
  enum {
    NONE,
    READ,
    WRITE,
    CUSTOM
  } type : 2;
  unsigned size : 30;
};

const unsigned ioctl_table_max = 500;
static ioctl_desc ioctl_table[ioctl_table_max];
static unsigned ioctl_table_size = 0;

// This can not be declared as a global, because references to struct_*_sz
// require a global initializer. And this table must be available before global
// initializers are run.
static void ioctl_table_fill() {
#define _(rq, tp, sz)                                  \
  CHECK(ioctl_table_size < ioctl_table_max);           \
  ioctl_table[ioctl_table_size].req = rq;              \
  ioctl_table[ioctl_table_size].type = ioctl_desc::tp; \
  ioctl_table[ioctl_table_size].size = sz;             \
  ++ioctl_table_size;

  _(0x0000540C, NONE, 0);                        // TIOCEXCL
  _(0x0000540D, NONE, 0);                        // TIOCNXCL
  _(0x0000540E, NONE, 0);                        // TIOCSCTTY
  _(0x0000540F, WRITE, pid_t_sz);                // TIOCGPGRP
  _(0x00005410, READ, pid_t_sz);                 // TIOCSPGRP
  _(0x00005411, WRITE, sizeof(int));             // TIOCOUTQ
  _(0x00005412, READ, sizeof(char));             // TIOCSTI
  _(0x00005413, WRITE, struct_winsize_sz);       // TIOCGWINSZ
  _(0x00005414, READ, struct_winsize_sz);        // TIOCSWINSZ
  _(0x00005415, WRITE, sizeof(int));             // TIOCMGET
  _(0x00005416, READ, sizeof(int));              // TIOCMBIS
  _(0x00005417, READ, sizeof(int));              // TIOCMBIC
  _(0x00005418, READ, sizeof(int));              // TIOCMSET
  _(0x0000541D, NONE, 0);                        // TIOCCONS
  _(0x00005420, READ, sizeof(int));              // TIOCPKT
  _(0x00005421, READ, sizeof(int));              // FIONBIO
  _(0x00005422, NONE, 0);                        // TIOCNOTTY
  _(0x00005423, READ, sizeof(int));              // TIOCSETD
  _(0x00005424, WRITE, sizeof(int));             // TIOCGETD
  _(0x00005450, NONE, 0);                        // FIONCLEX
  _(0x00005451, NONE, 0);                        // FIOCLEX
  _(0x00005452, READ, sizeof(int));              // FIOASYNC
  _(0x00008901, READ, sizeof(int));              // FIOSETOWN
  _(0x00008902, READ, sizeof(int));              // SIOCSPGRP
  _(0x00008903, WRITE, sizeof(int));             // FIOGETOWN
  _(0x00008904, WRITE, sizeof(int));             // SIOCGPGRP
  _(0x00008905, WRITE, sizeof(int));             // SIOCATMAR
  _(0x00008912, CUSTOM, 0);                      // SIOCGIFCONF
  _(0x00008913, WRITE, struct_ifreq_sz);         // SIOCGIFFLAGS
  _(0x00008914, READ, struct_ifreq_sz);          // SIOCSIFFLAGS
  _(0x00008915, WRITE, struct_ifreq_sz);         // SIOCGIFADDR
  _(0x00008916, READ, struct_ifreq_sz);          // SIOCSIFADDR
  _(0x00008917, WRITE, struct_ifreq_sz);         // SIOCGIFDSTADDR
  _(0x00008918, READ, struct_ifreq_sz);          // SIOCSIFDSTADDR
  _(0x00008919, WRITE, struct_ifreq_sz);         // SIOCGIFBRDADDR
  _(0x0000891A, READ, struct_ifreq_sz);          // SIOCSIFBRDADDR
  _(0x0000891B, WRITE, struct_ifreq_sz);         // SIOCGIFNETMASK
  _(0x0000891C, READ, struct_ifreq_sz);          // SIOCSIFNETMASK
  _(0x0000891D, WRITE, struct_ifreq_sz);         // SIOCGIFMETRIC
  _(0x0000891E, READ, struct_ifreq_sz);          // SIOCSIFMETRIC
  _(0x00008921, WRITE, struct_ifreq_sz);         // SIOCGIFMTU
  _(0x00008922, READ, struct_ifreq_sz);          // SIOCSIFMTU
  _(0x00008931, READ, struct_ifreq_sz);          // SIOCADDMULTI
  _(0x00008932, READ, struct_ifreq_sz);          // SIOCDELMULTI
  _(0x00008940, NONE, 0);                        // SIOCADDRTOLD
  _(0x00008941, NONE, 0);                        // SIOCDELRTOLD

#if (SANITIZER_LINUX && !SANITIZER_ANDROID) || SANITIZER_MAC
  _(0x000089E0, WRITE, struct_sioc_vif_req_sz);  // SIOCGETVIFCNT
  _(0x000089E1, WRITE, struct_sioc_sg_req_sz);   // SIOCGETSGCNT
#endif

#if SANITIZER_LINUX
  _(0x00000000, NONE, 0);                               // FDCLRPRM
  _(0x00000001, READ, struct_floppy_struct_sz);         // FDSETPRM
  _(0x00000002, READ, struct_floppy_struct_sz);         // FDDEFPRM
  _(0x00000003, WRITE, struct_floppy_struct_sz);        // FDGETPRM
  _(0x00000004, NONE, 0);                               // FDMSGON
  _(0x00000005, NONE, 0);                               // FDMSGOFF
  _(0x00000006, NONE, 0);                               // FDFMTBEG
  _(0x00000007, READ, struct_format_descr_sz);          // FDFMTTRK
  _(0x00000008, NONE, 0);                               // FDFMTEND
  _(0x0000000A, NONE, 0);                               // FDSETEMSGTRESH
  _(0x0000000B, NONE, 0);                               // FDFLUSH
  _(0x0000000C, READ, struct_floppy_max_errors_sz);     // FDSETMAXERRS
  _(0x0000000E, WRITE, struct_floppy_max_errors_sz);    // FDGETMAXERRS
  _(0x00000010, WRITE, 16);                             // FDGETDRVTYP
  _(0x00000014, READ, struct_floppy_drive_params_sz);   // FDSETDRVPRM
  _(0x00000015, WRITE, struct_floppy_drive_params_sz);  // FDGETDRVPRM
  _(0x00000016, WRITE, struct_floppy_drive_struct_sz);  // FDGETDRVSTAT
  _(0x00000017, WRITE, struct_floppy_drive_struct_sz);  // FDPOLLDRVSTAT
  _(0x00000018, NONE, 0);                               // FDRESET
  _(0x00000019, WRITE, struct_floppy_fdc_state_sz);     // FDGETFDCSTAT
  _(0x0000001B, NONE, 0);                               // FDWERRORCLR
  _(0x0000001C, WRITE, struct_floppy_write_errors_sz);  // FDWERRORGET
  _(0x0000001E, WRITE, struct_floppy_raw_cmd_sz);       // FDRAWCMD
  _(0x00000028, NONE, 0);                               // FDTWADDLE
  _(0x00000301, WRITE, struct_hd_geometry_sz);          // HDIO_GETGEO
  _(0x00000302, WRITE, sizeof(int));                    // HDIO_GET_UNMASKINTR
  _(0x00000304, WRITE, sizeof(int));                    // HDIO_GET_MULTCOUNT
  _(0x00000307, WRITE, struct_hd_driveid_sz);           // HDIO_GET_IDENTITY
  _(0x00000308, WRITE, sizeof(int));                    // HDIO_GET_KEEPSETTINGS
  _(0x00000309, WRITE, sizeof(int));                    // HDIO_GET_CHIPSET
  _(0x0000030A, WRITE, sizeof(int));                    // HDIO_GET_NOWERR
  _(0x0000030B, WRITE, sizeof(int));                    // HDIO_GET_DMA
  _(0x0000031F, WRITE, sizeof(int));                    // HDIO_DRIVE_CMD
  _(0x00000321, NONE, 0);                               // HDIO_SET_MULTCOUNT
  _(0x00000322, NONE, 0);                               // HDIO_SET_UNMASKINTR
  _(0x00000323, NONE, 0);                               // HDIO_SET_KEEPSETTINGS
  _(0x00000324, NONE, 0);                               // HDIO_SET_CHIPSET
  _(0x00000325, NONE, 0);                               // HDIO_SET_NOWERR
  _(0x00000326, NONE, 0);                               // HDIO_SET_DMA
  _(0x00000601, NONE, 0);                               // LPCHAR
  _(0x00000602, NONE, 0);                               // LPTIME
  _(0x00000604, NONE, 0);                               // LPABORT
  _(0x00000605, NONE, 0);                               // LPSETIRQ
  _(0x00000606, WRITE, sizeof(int));                    // LPGETIRQ
  _(0x00000608, NONE, 0);                               // LPWAIT
  _(0x00000609, NONE, 0);                               // LPCAREFUL
  _(0x0000060A, NONE, 0);                               // LPABORTOPEN
  _(0x0000060B, WRITE, sizeof(int));                    // LPGETSTATUS
  _(0x0000060C, NONE, 0);                               // LPRESET
  _(0x0000125D, READ, sizeof(int));                     // BLKROSET
  _(0x0000125E, WRITE, sizeof(int));                    // BLKROGET
  _(0x0000125F, NONE, 0);                               // BLKRRPART
  _(0x00001260, WRITE, sizeof(uptr));                   // BLKGETSIZE
  _(0x00001261, NONE, 0);                               // BLKFLSBUF
  _(0x00001262, NONE, 0);                               // BLKRASET
  _(0x00001263, WRITE, sizeof(int));                    // BLKRAGET
  _(0x00004300, NONE, 0);                               // SNDCTL_COPR_RESET
  _(0x00005000, NONE, 0);                               // SNDCTL_DSP_RESET
  _(0x00005001, NONE, 0);                               // SNDCTL_DSP_SYNC
  _(0x00005008, NONE, 0);                               // SNDCTL_DSP_POST
  _(0x0000500E, NONE, 0);                               // SNDCTL_DSP_NONBLOCK
  _(0x00005100, NONE, 0);                               // SNDCTL_SEQ_RESET
  _(0x00005101, NONE, 0);                               // SNDCTL_SEQ_SYNC
  _(0x00005111, NONE, 0);                               // SNDCTL_SEQ_PANIC
  _(0x00005301, NONE, 0);                               // CDROMPAUSE
  _(0x00005302, NONE, 0);                               // CDROMRESUME
  _(0x00005303, READ, struct_cdrom_msf_sz);             // CDROMPLAYMSF
  _(0x00005304, READ, struct_cdrom_ti_sz);              // CDROMPLAYTRKIND
  _(0x00005305, WRITE, struct_cdrom_tochdr_sz);         // CDROMREADTOCHDR
  _(0x00005306, WRITE, struct_cdrom_tocentry_sz);       // CDROMREADTOCENTRY
  _(0x00005307, NONE, 0);                               // CDROMSTOP
  _(0x00005308, NONE, 0);                               // CDROMSTART
  _(0x00005309, NONE, 0);                               // CDROMEJECT
  _(0x0000530A, READ, struct_cdrom_volctrl_sz);         // CDROMVOLCTRL
  _(0x0000530B, WRITE, struct_cdrom_subchnl_sz);        // CDROMSUBCHNL
  _(0x0000530C, READ, struct_cdrom_msf_sz);             // CDROMREADMODE2
  _(0x0000530D, READ, struct_cdrom_msf_sz);             // CDROMREADMODE1
  _(0x0000530E, READ, struct_cdrom_read_audio_sz);      // CDROMREADAUDIO
  _(0x0000530F, NONE, 0);                               // CDROMEJECT_SW
  _(0x00005310, WRITE, struct_cdrom_multisession_sz);   // CDROMMULTISESSION
  _(0x00005311, WRITE, 8);                              // CDROM_GET_UPC
  _(0x00005312, NONE, 0);                               // CDROMRESET
  _(0x00005313, WRITE, struct_cdrom_volctrl_sz);        // CDROMVOLREAD
  _(0x00005314, READ, struct_cdrom_msf_sz);             // CDROMREADRAW
  _(0x00005315, READ, struct_cdrom_msf_sz);             // CDROMREADCOOKED
  _(0x00005316, READ, struct_cdrom_msf_sz);             // CDROMSEEK
  // Conflicting request id.
  // _(0x00005382, NONE, 0);                      // CDROMAUDIOBUFSIZ
  // _(0x00005382, WRITE, 2 * sizeof(int));       // SCSI_IOCTL_GET_IDLUN
  _(0x00005383, NONE, 0);                         // SCSI_IOCTL_TAGGED_ENABLE
  _(0x00005384, NONE, 0);                         // SCSI_IOCTL_TAGGED_DISABLE
  _(0x00005385, READ, sizeof(int));               // SCSI_IOCTL_PROBE_HOST
  _(0x00005401, WRITE, struct_termios_sz);        // TCGETS
  _(0x00005402, READ, struct_termios_sz);         // TCSETS
  _(0x00005403, READ, struct_termios_sz);         // TCSETSW
  _(0x00005404, READ, struct_termios_sz);         // TCSETSF
  _(0x00005405, WRITE, struct_termio_sz);         // TCGETA
  _(0x00005406, READ, struct_termio_sz);          // TCSETA
  _(0x00005407, READ, struct_termio_sz);          // TCSETAW
  _(0x00005408, READ, struct_termio_sz);          // TCSETAF
  _(0x00005409, NONE, 0);                         // TCSBRK
  _(0x0000540A, NONE, 0);                         // TCXONC
  _(0x0000540B, NONE, 0);                         // TCFLSH
  // Conflicting request id.
  // _(0x00005402, NONE, 0);                         // SNDCTL_TMR_START
  // _(0x00005403, NONE, 0);                         // SNDCTL_TMR_STOP
  // _(0x00005404, NONE, 0);                         // SNDCTL_TMR_CONTINUE
  _(0x00005419, WRITE, sizeof(int));              // TIOCGSOFTCAR
  _(0x0000541A, READ, sizeof(int));               // TIOCSSOFTCAR
  _(0x0000541B, WRITE, sizeof(int));              // TIOCINQ
  _(0x0000541C, READ, sizeof(char));              // TIOCLINUX
  _(0x00005425, NONE, 0);                         // TCSBRKP
  _(0x00005453, NONE, 0);                         // TIOCSERCONFIG
  _(0x00005454, WRITE, sizeof(int));              // TIOCSERGWILD
  _(0x00005455, READ, sizeof(int));               // TIOCSERSWILD
  _(0x00005456, WRITE, struct_termios_sz);        // TIOCGLCKTRMIOS
  _(0x00005457, READ, struct_termios_sz);         // TIOCSLCKTRMIOS
  _(0x00005459, WRITE, sizeof(int));              // TIOCSERGETLSR
  _(0x00005470, NONE, 0);                         // TIOCSCCINI
  _(0x00005490, WRITE, sizeof(int));              // PPPIOCGFLAGS
  _(0x00005491, READ, sizeof(int));               // PPPIOCSFLAGS
  _(0x00005492, WRITE, sizeof(int));              // PPPIOCGASYNCMAP
  _(0x00005493, READ, sizeof(int));               // PPPIOCSASYNCMAP
  _(0x00005494, WRITE, sizeof(int));              // PPPIOCGUNIT
  _(0x00005495, READ, sizeof(int));               // PPPIOCSINPSIG
  _(0x00005497, READ, sizeof(int));               // PPPIOCSDEBUG
  _(0x00005498, WRITE, sizeof(int));              // PPPIOCGDEBUG
  _(0x0000549B, WRITE, sizeof(int) * 8);          // PPPIOCGXASYNCMAP
  _(0x0000549C, READ, sizeof(int) * 8);           // PPPIOCSXASYNCMAP
  _(0x0000549D, READ, sizeof(int));               // PPPIOCSMRU
  _(0x0000549E, READ, sizeof(int));               // PPPIOCRASYNCMAP
  _(0x0000549F, READ, sizeof(int));               // PPPIOCSMAXCID
  _(0x00005600, WRITE, sizeof(int));              // VT_OPENQRY
  _(0x00005601, WRITE, struct_vt_mode_sz);        // VT_GETMODE
  _(0x00005602, READ, struct_vt_mode_sz);         // VT_SETMODE
  _(0x00005603, WRITE, struct_vt_stat_sz);        // VT_GETSTATE
  _(0x00005604, NONE, 0);                         // VT_SENDSIG
  _(0x00005605, NONE, 0);                         // VT_RELDISP
  _(0x00005606, NONE, 0);                         // VT_ACTIVATE
  _(0x00005607, NONE, 0);                         // VT_WAITACTIVE
  _(0x00005608, NONE, 0);                         // VT_DISALLOCATE
  _(0x00005609, READ, struct_vt_sizes_sz);        // VT_RESIZE
  _(0x0000560A, READ, struct_vt_consize_sz);      // VT_RESIZEX
  _(0x00007314, NONE, 0);                         // STL_BINTR
  _(0x00007315, NONE, 0);                         // STL_BSTART
  _(0x00007316, NONE, 0);                         // STL_BSTOP
  _(0x00007317, NONE, 0);                         // STL_BRESET
  _(0x00008906, WRITE, timeval_sz);               // SIOCGSTAMP
  _(0x0000890B, READ, struct_rtentry_sz);         // SIOCADDRT
  _(0x0000890C, READ, struct_rtentry_sz);         // SIOCDELRT
  _(0x00008910, NONE, 0);                         // SIOCGIFNAME
  _(0x00008911, NONE, 0);                         // SIOCSIFLINK
  _(0x0000891F, WRITE, struct_ifreq_sz);          // SIOCGIFMEM
  _(0x00008920, READ, struct_ifreq_sz);           // SIOCSIFMEM
  _(0x00008923, WRITE, struct_ifreq_sz);          // OLD_SIOCGIFHWADDR
  _(0x00008924, READ, struct_ifreq_sz);           // SIOCSIFHWADDR
  _(0x00008925, WRITE, sizeof(int));              // SIOCGIFENCAP
  _(0x00008926, READ, sizeof(int));               // SIOCSIFENCAP
  _(0x00008927, WRITE, struct_ifreq_sz);          // SIOCGIFHWADDR
  _(0x00008929, NONE, 0);                         // SIOCGIFSLAVE
  _(0x00008930, NONE, 0);                         // SIOCSIFSLAVE
  _(0x00008950, READ, struct_arpreq_sz);          // SIOCDARP
  _(0x00008951, WRITE, struct_arpreq_sz);         // SIOCGARP
  _(0x00008952, READ, struct_arpreq_sz);          // SIOCSARP
  _(0x00008960, READ, struct_arpreq_sz);          // SIOCDRARP
  _(0x00008961, WRITE, struct_arpreq_sz);         // SIOCGRARP
  _(0x00008962, READ, struct_arpreq_sz);          // SIOCSRARP
  _(0x00008970, WRITE, struct_ifreq_sz);          // SIOCGIFMAP
  _(0x00008971, READ, struct_ifreq_sz);           // SIOCSIFMAP
  _(0x000089F0, WRITE, struct_ifreq_sz);          // SIOCDEVPLIP and EQL_ENSLAVE
  _(0x000089F1, WRITE, struct_ifreq_sz);          // EQL_EMANCIPATE
  _(0x000089F2, WRITE, struct_ifreq_sz);          // EQL_GETSLAVECFG
  _(0x000089F3, WRITE, struct_ifreq_sz);          // EQL_SETSLAVECFG
  _(0x000089F4, WRITE, struct_ifreq_sz);          // EQL_GETMASTRCFG
  _(0x000089F5, WRITE, struct_ifreq_sz);          // EQL_SETMASTRCFG
  _(0x00009000, READ, sizeof(int));               // DDIOCSDBG
  _(0x40045106, NONE, 0);                         // SNDCTL_SEQ_PERCMODE
  _(0x40045108, READ, sizeof(int));               // SNDCTL_SEQ_TESTMIDI
  _(0x40045109, READ, sizeof(int));               // SNDCTL_SEQ_RESETSAMPLES
  _(0x4004510D, READ, sizeof(int));               // SNDCTL_SEQ_THRESHOLD
  _(0x4004510F, READ, sizeof(int));               // SNDCTL_FM_4OP_ENABLE
  _(0x40045407, READ, sizeof(int));               // SNDCTL_TMR_METRONOME
  _(0x40045408, WRITE, sizeof(int));              // SNDCTL_TMR_SELECT
  _(0x40046602, READ, sizeof(int));               // EXT2_IOC_SETFLAGS
  _(0x40047602, READ, sizeof(int));               // EXT2_IOC_SETVERSION
  _(0x40085112, READ, struct_seq_event_rec_sz);   // SNDCTL_SEQ_OUTOFBAND
  _(0x40086D01, READ, struct_mtop_sz);            // MTIOCTOP
  _(0x40144304, READ, struct_copr_debug_buf_sz);  // SNDCTL_COPR_WDATA
  _(0x40144305, READ, struct_copr_debug_buf_sz);  // SNDCTL_COPR_WCODE
  _(0x40285107, READ, struct_sbi_instrument_sz);  // SNDCTL_FM_LOAD_INSTR
  _(0x4FA44308, READ, struct_copr_msg_sz);        // SNDCTL_COPR_SENDMSG
  _(0x80027501, WRITE, uid_t_sz);                 // SMB_IOC_GETMOUNTUID
  _(0x80044D00, WRITE, sizeof(int));              // SOUND_MIXER_READ_VOLUME
  _(0x80044D01, WRITE, sizeof(int));              // SOUND_MIXER_READ_BASS
  _(0x80044D02, WRITE, sizeof(int));              // SOUND_MIXER_READ_TREBLE
  _(0x80044D03, WRITE, sizeof(int));              // SOUND_MIXER_READ_SYNTH
  _(0x80044D04, WRITE, sizeof(int));              // SOUND_MIXER_READ_PCM
  _(0x80044D05, WRITE, sizeof(int));              // SOUND_MIXER_READ_SPEAKER
  _(0x80044D06, WRITE, sizeof(int));              // SOUND_MIXER_READ_LINE
  _(0x80044D07, WRITE, sizeof(int));              // SOUND_MIXER_READ_MIC
  _(0x80044D08, WRITE, sizeof(int));              // SOUND_MIXER_READ_CD
  _(0x80044D09, WRITE, sizeof(int));              // SOUND_MIXER_READ_IMIX
  _(0x80044D0A, WRITE, sizeof(int));              // SOUND_MIXER_READ_ALTPCM
  _(0x80044D0B, WRITE, sizeof(int));              // SOUND_MIXER_READ_RECLEV
  _(0x80044D0C, WRITE, sizeof(int));              // SOUND_MIXER_READ_IGAIN
  _(0x80044D0D, WRITE, sizeof(int));              // SOUND_MIXER_READ_OGAIN
  _(0x80044D0E, WRITE, sizeof(int));              // SOUND_MIXER_READ_LINE1
  _(0x80044D0F, WRITE, sizeof(int));              // SOUND_MIXER_READ_LINE2
  _(0x80044D10, WRITE, sizeof(int));              // SOUND_MIXER_READ_LINE3
  _(0x80044D1C, WRITE, sizeof(int));              // SOUND_MIXER_READ_MUTE
  _(0x80044D1D, WRITE, sizeof(int));              // SOUND_MIXER_READ_ENHANCE
  _(0x80044D1E, WRITE, sizeof(int));              // SOUND_MIXER_READ_LOUD
  _(0x80044DFB, WRITE, sizeof(int));              // SOUND_MIXER_READ_STEREODEVS
  _(0x80044DFC, WRITE, sizeof(int));              // SOUND_MIXER_READ_CAPS
  _(0x80044DFD, WRITE, sizeof(int));              // SOUND_MIXER_READ_RECMASK
  _(0x80044DFE, WRITE, sizeof(int));              // SOUND_MIXER_READ_DEVMASK
  _(0x80044DFF, WRITE, sizeof(int));              // SOUND_MIXER_READ_RECSRC
  _(0x80045002, WRITE, sizeof(int));              // SOUND_PCM_READ_RATE
  _(0x80045005, WRITE, sizeof(int));              // SOUND_PCM_READ_BITS
  _(0x80045006, WRITE, sizeof(int));              // SOUND_PCM_READ_CHANNELS
  _(0x80045007, WRITE, sizeof(int));              // SOUND_PCM_READ_FILTER
  _(0x8004500B, WRITE, sizeof(int));              // SNDCTL_DSP_GETFMTS
  _(0x80045104, WRITE, sizeof(int));              // SNDCTL_SEQ_GETOUTCOUNT
  _(0x80045105, WRITE, sizeof(int));              // SNDCTL_SEQ_GETINCOUNT
  _(0x8004510A, WRITE, sizeof(int));              // SNDCTL_SEQ_NRSYNTHS
  _(0x8004510B, WRITE, sizeof(int));              // SNDCTL_SEQ_NRMIDIS
  _(0x80046601, WRITE, sizeof(int));              // EXT2_IOC_GETFLAGS
  _(0x80046D03, WRITE, struct_mtpos_sz);          // MTIOCPOS
  _(0x80047601, WRITE, sizeof(int));              // EXT2_IOC_GETVERSION
  _(0x801C6D02, WRITE, struct_mtget_sz);          // MTIOCGET
  _(0x8FA44309, WRITE, struct_copr_msg_sz);       // SNDCTL_COPR_RCVMSG
  _(0xC0044D00, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_VOLUME
  _(0xC0044D01, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_BASS
  _(0xC0044D02, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_TREBLE
  _(0xC0044D03, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_SYNTH
  _(0xC0044D04, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_PCM
  _(0xC0044D05, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_SPEAKER
  _(0xC0044D06, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_LINE
  _(0xC0044D07, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_MIC
  _(0xC0044D08, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_CD
  _(0xC0044D09, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_IMIX
  _(0xC0044D0A, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_ALTPCM
  _(0xC0044D0B, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_RECLEV
  _(0xC0044D0C, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_IGAIN
  _(0xC0044D0D, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_OGAIN
  _(0xC0044D0E, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_LINE1
  _(0xC0044D0F, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_LINE2
  _(0xC0044D10, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_LINE3
  _(0xC0044D1C, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_MUTE
  _(0xC0044D1D, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_ENHANCE
  _(0xC0044D1E, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_LOUD
  _(0xC0044DFF, WRITE, sizeof(int));              // SOUND_MIXER_WRITE_RECSRC
  _(0xC0045002, WRITE, sizeof(int));              // SNDCTL_DSP_SPEED
  _(0xC0045003, WRITE, sizeof(int));              // SNDCTL_DSP_STEREO
  _(0xC0045004, WRITE, sizeof(int));              // SNDCTL_DSP_GETBLKSIZE
  _(0xC0045005, WRITE, sizeof(int));              // SNDCTL_DSP_SETFMT
  _(0xC0045006, WRITE, sizeof(int));              // SOUND_PCM_WRITE_CHANNELS
  _(0xC0045007, WRITE, sizeof(int));              // SOUND_PCM_WRITE_FILTER
  _(0xC0045009, WRITE, sizeof(int));              // SNDCTL_DSP_SUBDIVIDE
  _(0xC004500A, WRITE, sizeof(int));              // SNDCTL_DSP_SETFRAGMENT
  _(0xC0045103, WRITE, sizeof(int));              // SNDCTL_SEQ_CTRLRATE
  _(0xC004510E, WRITE, sizeof(int));              // SNDCTL_SYNTH_MEMAVL
  _(0xC0045401, WRITE, sizeof(int));              // SNDCTL_TMR_TIMEBASE
  _(0xC0045405, WRITE, sizeof(int));              // SNDCTL_TMR_TEMPO
  _(0xC0045406, WRITE, sizeof(int));              // SNDCTL_TMR_SOURCE
  _(0xC0046D00, WRITE, sizeof(int));              // SNDCTL_MIDI_PRETIME
  _(0xC0046D01, READ, sizeof(int));               // SNDCTL_MIDI_MPUMODE
  _(0xC0144302, WRITE, struct_copr_debug_buf_sz);  // SNDCTL_COPR_RDATA
  _(0xC0144303, WRITE, struct_copr_debug_buf_sz);  // SNDCTL_COPR_RCODE
  _(0xC0144306, WRITE, struct_copr_debug_buf_sz);  // SNDCTL_COPR_RUN
  _(0xC0144307, WRITE, struct_copr_debug_buf_sz);  // SNDCTL_COPR_HALT
  _(0xC074510C, WRITE, struct_midi_info_sz);       // SNDCTL_MIDI_INFO
  _(0xC08C5102, WRITE, struct_synth_info_sz);      // SNDCTL_SYNTH_INFO
  _(0xCFB04301, READ, struct_copr_buffer_sz);      // SNDCTL_COPR_LOAD
#endif

#if SANITIZER_LINUX && !SANITIZER_ANDROID
  _(0x00005499, WRITE, struct_ppp_stats_sz);  // PPPIOCGSTAT

  _(0x00004B2F, NONE, 0);                         // KIOCSOUND
  _(0x00004B30, NONE, 0);                         // KDMKTONE
  _(0x00004B31, WRITE, 1);                        // KDGETLED
  _(0x00004B32, NONE, 0);                         // KDSETLED
  _(0x00004B33, WRITE, 1);                        // KDGKBTYPE
  _(0x00004B34, NONE, 0);                         // KDADDIO
  _(0x00004B35, NONE, 0);                         // KDDELIO
  _(0x00004B36, NONE, 0);                         // KDENABIO
  _(0x00004B37, NONE, 0);                         // KDDISABIO
  _(0x00004B3A, NONE, 0);                         // KDSETMODE
  _(0x00004B3B, WRITE, sizeof(int));              // KDGETMODE
  _(0x00004B3C, NONE, 0);                         // KDMAPDISP
  _(0x00004B3D, NONE, 0);                         // KDUNMAPDISP
  _(0x00004B40, WRITE, e_tabsz);                  // GIO_SCRNMAP
  _(0x00004B41, READ, e_tabsz);                   // PIO_SCRNMAP
  _(0x00004B44, WRITE, sizeof(int));              // KDGKBMODE
  _(0x00004B45, NONE, 0);                         // KDSKBMODE
  _(0x00004B46, WRITE, struct_kbentry_sz);        // KDGKBENT
  _(0x00004B47, READ, struct_kbentry_sz);         // KDSKBENT
  _(0x00004B48, WRITE, struct_kbsentry_sz);       // KDGKBSENT
  _(0x00004B49, READ, struct_kbsentry_sz);        // KDSKBSENT
  _(0x00004B4A, WRITE, struct_kbdiacrs_sz);       // KDGKBDIACR
  _(0x00004B4B, READ, struct_kbdiacrs_sz);        // KDSKBDIACR
  _(0x00004B4C, WRITE, struct_kbkeycode_sz);      // KDGETKEYCODE
  _(0x00004B4D, READ, struct_kbkeycode_sz);       // KDSETKEYCODE
  _(0x00004B4E, NONE, 0);                         // KDSIGACCEPT
  _(0x00004B60, WRITE, 8192);                     // GIO_FONT
  _(0x00004B61, READ, 8192);                      // PIO_FONT
  _(0x00004B62, WRITE, sizeof(int));              // KDGKBMETA
  _(0x00004B63, NONE, 0);                         // KDSKBMETA
  _(0x00004B64, WRITE, sizeof(int));              // KDGKBLED
  _(0x00004B65, NONE, 0);                         // KDSKBLED
  _(0x00004B66, WRITE, struct_unimapdesc_sz);     // GIO_UNIMAP
  _(0x00004B67, READ, struct_unimapdesc_sz);      // PIO_UNIMAP
  _(0x00004B68, READ, struct_unimapinit_sz);      // PIO_UNIMAPCLR
  _(0x00004B69, WRITE, sizeof(short) * e_tabsz);  // GIO_UNISCRNMAP
  _(0x00004B6A, READ, sizeof(short) * e_tabsz);   // PIO_UNISCRNMAP
  _(0x00004B70, WRITE, 48);                       // GIO_CMAP
  _(0x00004B71, NONE, 0);                         // PIO_CMAP

  // Missing struct definition on Android.
  _(0x0000541E, WRITE, struct_serial_struct_sz);            // TIOCGSERIAL
  _(0x0000541F, READ, struct_serial_struct_sz);             // TIOCSSERIAL
  _(0x0000545A, WRITE, struct_serial_multiport_struct_sz);  // TIOCSERGETMULTI
  _(0x0000545B, READ, struct_serial_multiport_struct_sz);   // TIOCSERSETMULTI
  _(0x00005471, READ, struct_scc_modem_sz);                 // TIOCCHANINI
  _(0x00005474, WRITE, struct_scc_stat_sz);                 // TIOCSCCSTAT

  // The following ioctl requests are shared between AX25, IPX, netrom and
  // mrouted.
  // _(0x000089E0, READ, sizeof(char));                  // SIOCAIPXITFCRT
  // _(0x000089E0, READ, struct_sockaddr_ax25_sz);       // SIOCAX25GETUID
  // _(0x000089E0, WRITE, struct_nr_parms_struct_sz);    // SIOCNRGETPARMS
  // _(0x000089E1, READ, sizeof(char));                  // SIOCAIPXPRISLT
  // _(0x000089E1, READ, struct_nr_parms_struct_sz);     // SIOCNRSETPARMS
  // _(0x000089E1, READ, struct_sockaddr_ax25_sz);       // SIOCAX25ADDUID
  // _(0x000089E2, NONE, 0);                             // SIOCNRDECOBS
  // _(0x000089E2, READ, struct_sockaddr_ax25_sz);       // SIOCAX25DELUID
  // _(0x000089E2, WRITE, struct_ipx_config_data_sz);    // SIOCIPXCFGDATA
  // _(0x000089E3, READ, sizeof(int));                   // SIOCAX25NOUID
  // _(0x000089E3, READ, sizeof(int));                   // SIOCNRRTCTL
  // _(0x000089E4, READ, sizeof(int));                   // SIOCAX25DIGCTL
  // _(0x000089E5, WRITE, struct_ax25_parms_struct_sz);  // SIOCAX25GETPARMS
  // _(0x000089E6, READ, struct_ax25_parms_struct_sz);   // SIOCAX25SETPARMS

  _(0x00435901, WRITE, struct_cyclades_monitor_sz);  // CYGETMON
  _(0x00435902, WRITE, sizeof(int));                 // CYGETTHRESH
  _(0x00435903, NONE, 0);                            // CYSETTHRESH
  _(0x00435904, WRITE, sizeof(int));                 // CYGETDEFTHRESH
  _(0x00435905, NONE, 0);                            // CYSETDEFTHRESH
  _(0x00435906, WRITE, sizeof(int));                 // CYGETTIMEOUT
  _(0x00435907, NONE, 0);                            // CYSETTIMEOUT
  _(0x00435908, WRITE, sizeof(int));                 // CYGETDEFTIMEOUT
  _(0x00435909, NONE, 0);                            // CYSETDEFTIMEOUT

  // Missing definition of mtconfiginfo
  _(0x40206D05, READ, struct_mtconfiginfo_sz);   // MTIOCSETCONFIG
  _(0x80206D04, WRITE, struct_mtconfiginfo_sz);  // MTIOCGETCONFIG

  _(0x800C500C, WRITE, struct_audio_buf_info_sz);  // SNDCTL_DSP_GETOSPACE
  _(0x800C500D, WRITE, struct_audio_buf_info_sz);  // SNDCTL_DSP_GETISPACE
#endif
#undef _
  // FIXME: missing USB ioctls (EVIOxx...x).
}

static bool ioctl_initialized = false;

struct ioctl_desc_compare {
  bool operator()(const ioctl_desc& left, const ioctl_desc& right) const {
    return left.req < right.req;
  }
};

static void ioctl_init() {
  ioctl_table_fill();
  InternalSort(&ioctl_table, ioctl_table_size, ioctl_desc_compare());

  for (unsigned i = 0; i < ioctl_table_size - 1; ++i) {
    if (ioctl_table[i].req >= ioctl_table[i + 1].req) {
      Printf("Duplicate or unsorted ioctl request id %x >= %x\n",
             ioctl_table[i].req, ioctl_table[i + 1].req);
      Die();
    }
  }

  ioctl_initialized = true;
}

static const ioctl_desc *ioctl_lookup(unsigned req) {
  int left = 0;
  int right = sizeof(ioctl_table) / sizeof(*ioctl_table);
  while (left < right) {
    int mid = (left + right) / 2;
    if (ioctl_table[mid].req < req)
      left = mid + 1;
    else
      right = mid;
  }
  if (left == right && ioctl_table[left].req == req)
    return ioctl_table + left;
  else
    return 0;
}

static void ioctl_common_pre(void *ctx, const ioctl_desc *desc, int d,
                             unsigned request, void *arg) {
  if (desc->type == ioctl_desc::READ)
    COMMON_INTERCEPTOR_READ_RANGE(ctx, arg, desc->size);
  if (desc->type != ioctl_desc::CUSTOM)
    return;
  switch (request) {
    case 0x00008912: {  // SIOCGIFCONF
      struct __sanitizer_ifconf *ifc = (__sanitizer_ifconf *)arg;
      COMMON_INTERCEPTOR_READ_RANGE(ctx, &ifc->ifc_len, sizeof(ifc->ifc_len));
      break;
    }
  }
  return;
}

static void ioctl_common_post(void *ctx, const ioctl_desc *desc, int res, int d,
                              unsigned request, void *arg) {
  if (desc->type == ioctl_desc::WRITE) {
    // FIXME: add verbose output
    COMMON_INTERCEPTOR_WRITE_RANGE(ctx, arg, desc->size);
  }
  if (desc->type != ioctl_desc::CUSTOM)
    return;
  switch (request) {
    case 0x00008912: {  // SIOCGIFCONF
      struct __sanitizer_ifconf *ifc = (__sanitizer_ifconf *)arg;
      COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ifc->ifc_ifcu.ifcu_req, ifc->ifc_len);
      break;
    }
  }
  return;
}
