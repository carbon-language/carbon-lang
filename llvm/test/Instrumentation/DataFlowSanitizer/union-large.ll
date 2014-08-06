; RUN: opt < %s -dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Check that we use dfsan_union in large functions instead of __dfsan_union.

; CHECK-LABEL: @"dfs$foo"
define i32 @foo(i32 %a, i32 %b) {
bb0:
  br label %bb1

bb1:
  br label %bb2

bb2:
  br label %bb3

bb3:
  br label %bb4

bb4:
  br label %bb5

bb5:
  br label %bb6

bb6:
  br label %bb7

bb7:
  br label %bb8

bb8:
  br label %bb9

bb9:
  br label %bb10

bb10:
  br label %bb11

bb11:
  br label %bb12

bb12:
  br label %bb13

bb13:
  br label %bb14

bb14:
  br label %bb15

bb15:
  br label %bb16

bb16:
  br label %bb17

bb17:
  br label %bb18

bb18:
  br label %bb19

bb19:
  br label %bb20

bb20:
  br label %bb21

bb21:
  br label %bb22

bb22:
  br label %bb23

bb23:
  br label %bb24

bb24:
  br label %bb25

bb25:
  br label %bb26

bb26:
  br label %bb27

bb27:
  br label %bb28

bb28:
  br label %bb29

bb29:
  br label %bb30

bb30:
  br label %bb31

bb31:
  br label %bb32

bb32:
  br label %bb33

bb33:
  br label %bb34

bb34:
  br label %bb35

bb35:
  br label %bb36

bb36:
  br label %bb37

bb37:
  br label %bb38

bb38:
  br label %bb39

bb39:
  br label %bb40

bb40:
  br label %bb41

bb41:
  br label %bb42

bb42:
  br label %bb43

bb43:
  br label %bb44

bb44:
  br label %bb45

bb45:
  br label %bb46

bb46:
  br label %bb47

bb47:
  br label %bb48

bb48:
  br label %bb49

bb49:
  br label %bb50

bb50:
  br label %bb51

bb51:
  br label %bb52

bb52:
  br label %bb53

bb53:
  br label %bb54

bb54:
  br label %bb55

bb55:
  br label %bb56

bb56:
  br label %bb57

bb57:
  br label %bb58

bb58:
  br label %bb59

bb59:
  br label %bb60

bb60:
  br label %bb61

bb61:
  br label %bb62

bb62:
  br label %bb63

bb63:
  br label %bb64

bb64:
  br label %bb65

bb65:
  br label %bb66

bb66:
  br label %bb67

bb67:
  br label %bb68

bb68:
  br label %bb69

bb69:
  br label %bb70

bb70:
  br label %bb71

bb71:
  br label %bb72

bb72:
  br label %bb73

bb73:
  br label %bb74

bb74:
  br label %bb75

bb75:
  br label %bb76

bb76:
  br label %bb77

bb77:
  br label %bb78

bb78:
  br label %bb79

bb79:
  br label %bb80

bb80:
  br label %bb81

bb81:
  br label %bb82

bb82:
  br label %bb83

bb83:
  br label %bb84

bb84:
  br label %bb85

bb85:
  br label %bb86

bb86:
  br label %bb87

bb87:
  br label %bb88

bb88:
  br label %bb89

bb89:
  br label %bb90

bb90:
  br label %bb91

bb91:
  br label %bb92

bb92:
  br label %bb93

bb93:
  br label %bb94

bb94:
  br label %bb95

bb95:
  br label %bb96

bb96:
  br label %bb97

bb97:
  br label %bb98

bb98:
  br label %bb99

bb99:
  br label %bb100

bb100:
  br label %bb101

bb101:
  br label %bb102

bb102:
  br label %bb103

bb103:
  br label %bb104

bb104:
  br label %bb105

bb105:
  br label %bb106

bb106:
  br label %bb107

bb107:
  br label %bb108

bb108:
  br label %bb109

bb109:
  br label %bb110

bb110:
  br label %bb111

bb111:
  br label %bb112

bb112:
  br label %bb113

bb113:
  br label %bb114

bb114:
  br label %bb115

bb115:
  br label %bb116

bb116:
  br label %bb117

bb117:
  br label %bb118

bb118:
  br label %bb119

bb119:
  br label %bb120

bb120:
  br label %bb121

bb121:
  br label %bb122

bb122:
  br label %bb123

bb123:
  br label %bb124

bb124:
  br label %bb125

bb125:
  br label %bb126

bb126:
  br label %bb127

bb127:
  br label %bb128

bb128:
  br label %bb129

bb129:
  br label %bb130

bb130:
  br label %bb131

bb131:
  br label %bb132

bb132:
  br label %bb133

bb133:
  br label %bb134

bb134:
  br label %bb135

bb135:
  br label %bb136

bb136:
  br label %bb137

bb137:
  br label %bb138

bb138:
  br label %bb139

bb139:
  br label %bb140

bb140:
  br label %bb141

bb141:
  br label %bb142

bb142:
  br label %bb143

bb143:
  br label %bb144

bb144:
  br label %bb145

bb145:
  br label %bb146

bb146:
  br label %bb147

bb147:
  br label %bb148

bb148:
  br label %bb149

bb149:
  br label %bb150

bb150:
  br label %bb151

bb151:
  br label %bb152

bb152:
  br label %bb153

bb153:
  br label %bb154

bb154:
  br label %bb155

bb155:
  br label %bb156

bb156:
  br label %bb157

bb157:
  br label %bb158

bb158:
  br label %bb159

bb159:
  br label %bb160

bb160:
  br label %bb161

bb161:
  br label %bb162

bb162:
  br label %bb163

bb163:
  br label %bb164

bb164:
  br label %bb165

bb165:
  br label %bb166

bb166:
  br label %bb167

bb167:
  br label %bb168

bb168:
  br label %bb169

bb169:
  br label %bb170

bb170:
  br label %bb171

bb171:
  br label %bb172

bb172:
  br label %bb173

bb173:
  br label %bb174

bb174:
  br label %bb175

bb175:
  br label %bb176

bb176:
  br label %bb177

bb177:
  br label %bb178

bb178:
  br label %bb179

bb179:
  br label %bb180

bb180:
  br label %bb181

bb181:
  br label %bb182

bb182:
  br label %bb183

bb183:
  br label %bb184

bb184:
  br label %bb185

bb185:
  br label %bb186

bb186:
  br label %bb187

bb187:
  br label %bb188

bb188:
  br label %bb189

bb189:
  br label %bb190

bb190:
  br label %bb191

bb191:
  br label %bb192

bb192:
  br label %bb193

bb193:
  br label %bb194

bb194:
  br label %bb195

bb195:
  br label %bb196

bb196:
  br label %bb197

bb197:
  br label %bb198

bb198:
  br label %bb199

bb199:
  br label %bb200

bb200:
  br label %bb201

bb201:
  br label %bb202

bb202:
  br label %bb203

bb203:
  br label %bb204

bb204:
  br label %bb205

bb205:
  br label %bb206

bb206:
  br label %bb207

bb207:
  br label %bb208

bb208:
  br label %bb209

bb209:
  br label %bb210

bb210:
  br label %bb211

bb211:
  br label %bb212

bb212:
  br label %bb213

bb213:
  br label %bb214

bb214:
  br label %bb215

bb215:
  br label %bb216

bb216:
  br label %bb217

bb217:
  br label %bb218

bb218:
  br label %bb219

bb219:
  br label %bb220

bb220:
  br label %bb221

bb221:
  br label %bb222

bb222:
  br label %bb223

bb223:
  br label %bb224

bb224:
  br label %bb225

bb225:
  br label %bb226

bb226:
  br label %bb227

bb227:
  br label %bb228

bb228:
  br label %bb229

bb229:
  br label %bb230

bb230:
  br label %bb231

bb231:
  br label %bb232

bb232:
  br label %bb233

bb233:
  br label %bb234

bb234:
  br label %bb235

bb235:
  br label %bb236

bb236:
  br label %bb237

bb237:
  br label %bb238

bb238:
  br label %bb239

bb239:
  br label %bb240

bb240:
  br label %bb241

bb241:
  br label %bb242

bb242:
  br label %bb243

bb243:
  br label %bb244

bb244:
  br label %bb245

bb245:
  br label %bb246

bb246:
  br label %bb247

bb247:
  br label %bb248

bb248:
  br label %bb249

bb249:
  br label %bb250

bb250:
  br label %bb251

bb251:
  br label %bb252

bb252:
  br label %bb253

bb253:
  br label %bb254

bb254:
  br label %bb255

bb255:
  br label %bb256

bb256:
  br label %bb257

bb257:
  br label %bb258

bb258:
  br label %bb259

bb259:
  br label %bb260

bb260:
  br label %bb261

bb261:
  br label %bb262

bb262:
  br label %bb263

bb263:
  br label %bb264

bb264:
  br label %bb265

bb265:
  br label %bb266

bb266:
  br label %bb267

bb267:
  br label %bb268

bb268:
  br label %bb269

bb269:
  br label %bb270

bb270:
  br label %bb271

bb271:
  br label %bb272

bb272:
  br label %bb273

bb273:
  br label %bb274

bb274:
  br label %bb275

bb275:
  br label %bb276

bb276:
  br label %bb277

bb277:
  br label %bb278

bb278:
  br label %bb279

bb279:
  br label %bb280

bb280:
  br label %bb281

bb281:
  br label %bb282

bb282:
  br label %bb283

bb283:
  br label %bb284

bb284:
  br label %bb285

bb285:
  br label %bb286

bb286:
  br label %bb287

bb287:
  br label %bb288

bb288:
  br label %bb289

bb289:
  br label %bb290

bb290:
  br label %bb291

bb291:
  br label %bb292

bb292:
  br label %bb293

bb293:
  br label %bb294

bb294:
  br label %bb295

bb295:
  br label %bb296

bb296:
  br label %bb297

bb297:
  br label %bb298

bb298:
  br label %bb299

bb299:
  br label %bb300

bb300:
  br label %bb301

bb301:
  br label %bb302

bb302:
  br label %bb303

bb303:
  br label %bb304

bb304:
  br label %bb305

bb305:
  br label %bb306

bb306:
  br label %bb307

bb307:
  br label %bb308

bb308:
  br label %bb309

bb309:
  br label %bb310

bb310:
  br label %bb311

bb311:
  br label %bb312

bb312:
  br label %bb313

bb313:
  br label %bb314

bb314:
  br label %bb315

bb315:
  br label %bb316

bb316:
  br label %bb317

bb317:
  br label %bb318

bb318:
  br label %bb319

bb319:
  br label %bb320

bb320:
  br label %bb321

bb321:
  br label %bb322

bb322:
  br label %bb323

bb323:
  br label %bb324

bb324:
  br label %bb325

bb325:
  br label %bb326

bb326:
  br label %bb327

bb327:
  br label %bb328

bb328:
  br label %bb329

bb329:
  br label %bb330

bb330:
  br label %bb331

bb331:
  br label %bb332

bb332:
  br label %bb333

bb333:
  br label %bb334

bb334:
  br label %bb335

bb335:
  br label %bb336

bb336:
  br label %bb337

bb337:
  br label %bb338

bb338:
  br label %bb339

bb339:
  br label %bb340

bb340:
  br label %bb341

bb341:
  br label %bb342

bb342:
  br label %bb343

bb343:
  br label %bb344

bb344:
  br label %bb345

bb345:
  br label %bb346

bb346:
  br label %bb347

bb347:
  br label %bb348

bb348:
  br label %bb349

bb349:
  br label %bb350

bb350:
  br label %bb351

bb351:
  br label %bb352

bb352:
  br label %bb353

bb353:
  br label %bb354

bb354:
  br label %bb355

bb355:
  br label %bb356

bb356:
  br label %bb357

bb357:
  br label %bb358

bb358:
  br label %bb359

bb359:
  br label %bb360

bb360:
  br label %bb361

bb361:
  br label %bb362

bb362:
  br label %bb363

bb363:
  br label %bb364

bb364:
  br label %bb365

bb365:
  br label %bb366

bb366:
  br label %bb367

bb367:
  br label %bb368

bb368:
  br label %bb369

bb369:
  br label %bb370

bb370:
  br label %bb371

bb371:
  br label %bb372

bb372:
  br label %bb373

bb373:
  br label %bb374

bb374:
  br label %bb375

bb375:
  br label %bb376

bb376:
  br label %bb377

bb377:
  br label %bb378

bb378:
  br label %bb379

bb379:
  br label %bb380

bb380:
  br label %bb381

bb381:
  br label %bb382

bb382:
  br label %bb383

bb383:
  br label %bb384

bb384:
  br label %bb385

bb385:
  br label %bb386

bb386:
  br label %bb387

bb387:
  br label %bb388

bb388:
  br label %bb389

bb389:
  br label %bb390

bb390:
  br label %bb391

bb391:
  br label %bb392

bb392:
  br label %bb393

bb393:
  br label %bb394

bb394:
  br label %bb395

bb395:
  br label %bb396

bb396:
  br label %bb397

bb397:
  br label %bb398

bb398:
  br label %bb399

bb399:
  br label %bb400

bb400:
  br label %bb401

bb401:
  br label %bb402

bb402:
  br label %bb403

bb403:
  br label %bb404

bb404:
  br label %bb405

bb405:
  br label %bb406

bb406:
  br label %bb407

bb407:
  br label %bb408

bb408:
  br label %bb409

bb409:
  br label %bb410

bb410:
  br label %bb411

bb411:
  br label %bb412

bb412:
  br label %bb413

bb413:
  br label %bb414

bb414:
  br label %bb415

bb415:
  br label %bb416

bb416:
  br label %bb417

bb417:
  br label %bb418

bb418:
  br label %bb419

bb419:
  br label %bb420

bb420:
  br label %bb421

bb421:
  br label %bb422

bb422:
  br label %bb423

bb423:
  br label %bb424

bb424:
  br label %bb425

bb425:
  br label %bb426

bb426:
  br label %bb427

bb427:
  br label %bb428

bb428:
  br label %bb429

bb429:
  br label %bb430

bb430:
  br label %bb431

bb431:
  br label %bb432

bb432:
  br label %bb433

bb433:
  br label %bb434

bb434:
  br label %bb435

bb435:
  br label %bb436

bb436:
  br label %bb437

bb437:
  br label %bb438

bb438:
  br label %bb439

bb439:
  br label %bb440

bb440:
  br label %bb441

bb441:
  br label %bb442

bb442:
  br label %bb443

bb443:
  br label %bb444

bb444:
  br label %bb445

bb445:
  br label %bb446

bb446:
  br label %bb447

bb447:
  br label %bb448

bb448:
  br label %bb449

bb449:
  br label %bb450

bb450:
  br label %bb451

bb451:
  br label %bb452

bb452:
  br label %bb453

bb453:
  br label %bb454

bb454:
  br label %bb455

bb455:
  br label %bb456

bb456:
  br label %bb457

bb457:
  br label %bb458

bb458:
  br label %bb459

bb459:
  br label %bb460

bb460:
  br label %bb461

bb461:
  br label %bb462

bb462:
  br label %bb463

bb463:
  br label %bb464

bb464:
  br label %bb465

bb465:
  br label %bb466

bb466:
  br label %bb467

bb467:
  br label %bb468

bb468:
  br label %bb469

bb469:
  br label %bb470

bb470:
  br label %bb471

bb471:
  br label %bb472

bb472:
  br label %bb473

bb473:
  br label %bb474

bb474:
  br label %bb475

bb475:
  br label %bb476

bb476:
  br label %bb477

bb477:
  br label %bb478

bb478:
  br label %bb479

bb479:
  br label %bb480

bb480:
  br label %bb481

bb481:
  br label %bb482

bb482:
  br label %bb483

bb483:
  br label %bb484

bb484:
  br label %bb485

bb485:
  br label %bb486

bb486:
  br label %bb487

bb487:
  br label %bb488

bb488:
  br label %bb489

bb489:
  br label %bb490

bb490:
  br label %bb491

bb491:
  br label %bb492

bb492:
  br label %bb493

bb493:
  br label %bb494

bb494:
  br label %bb495

bb495:
  br label %bb496

bb496:
  br label %bb497

bb497:
  br label %bb498

bb498:
  br label %bb499

bb499:
  br label %bb500

bb500:
  br label %bb501

bb501:
  br label %bb502

bb502:
  br label %bb503

bb503:
  br label %bb504

bb504:
  br label %bb505

bb505:
  br label %bb506

bb506:
  br label %bb507

bb507:
  br label %bb508

bb508:
  br label %bb509

bb509:
  br label %bb510

bb510:
  br label %bb511

bb511:
  br label %bb512

bb512:
  br label %bb513

bb513:
  br label %bb514

bb514:
  br label %bb515

bb515:
  br label %bb516

bb516:
  br label %bb517

bb517:
  br label %bb518

bb518:
  br label %bb519

bb519:
  br label %bb520

bb520:
  br label %bb521

bb521:
  br label %bb522

bb522:
  br label %bb523

bb523:
  br label %bb524

bb524:
  br label %bb525

bb525:
  br label %bb526

bb526:
  br label %bb527

bb527:
  br label %bb528

bb528:
  br label %bb529

bb529:
  br label %bb530

bb530:
  br label %bb531

bb531:
  br label %bb532

bb532:
  br label %bb533

bb533:
  br label %bb534

bb534:
  br label %bb535

bb535:
  br label %bb536

bb536:
  br label %bb537

bb537:
  br label %bb538

bb538:
  br label %bb539

bb539:
  br label %bb540

bb540:
  br label %bb541

bb541:
  br label %bb542

bb542:
  br label %bb543

bb543:
  br label %bb544

bb544:
  br label %bb545

bb545:
  br label %bb546

bb546:
  br label %bb547

bb547:
  br label %bb548

bb548:
  br label %bb549

bb549:
  br label %bb550

bb550:
  br label %bb551

bb551:
  br label %bb552

bb552:
  br label %bb553

bb553:
  br label %bb554

bb554:
  br label %bb555

bb555:
  br label %bb556

bb556:
  br label %bb557

bb557:
  br label %bb558

bb558:
  br label %bb559

bb559:
  br label %bb560

bb560:
  br label %bb561

bb561:
  br label %bb562

bb562:
  br label %bb563

bb563:
  br label %bb564

bb564:
  br label %bb565

bb565:
  br label %bb566

bb566:
  br label %bb567

bb567:
  br label %bb568

bb568:
  br label %bb569

bb569:
  br label %bb570

bb570:
  br label %bb571

bb571:
  br label %bb572

bb572:
  br label %bb573

bb573:
  br label %bb574

bb574:
  br label %bb575

bb575:
  br label %bb576

bb576:
  br label %bb577

bb577:
  br label %bb578

bb578:
  br label %bb579

bb579:
  br label %bb580

bb580:
  br label %bb581

bb581:
  br label %bb582

bb582:
  br label %bb583

bb583:
  br label %bb584

bb584:
  br label %bb585

bb585:
  br label %bb586

bb586:
  br label %bb587

bb587:
  br label %bb588

bb588:
  br label %bb589

bb589:
  br label %bb590

bb590:
  br label %bb591

bb591:
  br label %bb592

bb592:
  br label %bb593

bb593:
  br label %bb594

bb594:
  br label %bb595

bb595:
  br label %bb596

bb596:
  br label %bb597

bb597:
  br label %bb598

bb598:
  br label %bb599

bb599:
  br label %bb600

bb600:
  br label %bb601

bb601:
  br label %bb602

bb602:
  br label %bb603

bb603:
  br label %bb604

bb604:
  br label %bb605

bb605:
  br label %bb606

bb606:
  br label %bb607

bb607:
  br label %bb608

bb608:
  br label %bb609

bb609:
  br label %bb610

bb610:
  br label %bb611

bb611:
  br label %bb612

bb612:
  br label %bb613

bb613:
  br label %bb614

bb614:
  br label %bb615

bb615:
  br label %bb616

bb616:
  br label %bb617

bb617:
  br label %bb618

bb618:
  br label %bb619

bb619:
  br label %bb620

bb620:
  br label %bb621

bb621:
  br label %bb622

bb622:
  br label %bb623

bb623:
  br label %bb624

bb624:
  br label %bb625

bb625:
  br label %bb626

bb626:
  br label %bb627

bb627:
  br label %bb628

bb628:
  br label %bb629

bb629:
  br label %bb630

bb630:
  br label %bb631

bb631:
  br label %bb632

bb632:
  br label %bb633

bb633:
  br label %bb634

bb634:
  br label %bb635

bb635:
  br label %bb636

bb636:
  br label %bb637

bb637:
  br label %bb638

bb638:
  br label %bb639

bb639:
  br label %bb640

bb640:
  br label %bb641

bb641:
  br label %bb642

bb642:
  br label %bb643

bb643:
  br label %bb644

bb644:
  br label %bb645

bb645:
  br label %bb646

bb646:
  br label %bb647

bb647:
  br label %bb648

bb648:
  br label %bb649

bb649:
  br label %bb650

bb650:
  br label %bb651

bb651:
  br label %bb652

bb652:
  br label %bb653

bb653:
  br label %bb654

bb654:
  br label %bb655

bb655:
  br label %bb656

bb656:
  br label %bb657

bb657:
  br label %bb658

bb658:
  br label %bb659

bb659:
  br label %bb660

bb660:
  br label %bb661

bb661:
  br label %bb662

bb662:
  br label %bb663

bb663:
  br label %bb664

bb664:
  br label %bb665

bb665:
  br label %bb666

bb666:
  br label %bb667

bb667:
  br label %bb668

bb668:
  br label %bb669

bb669:
  br label %bb670

bb670:
  br label %bb671

bb671:
  br label %bb672

bb672:
  br label %bb673

bb673:
  br label %bb674

bb674:
  br label %bb675

bb675:
  br label %bb676

bb676:
  br label %bb677

bb677:
  br label %bb678

bb678:
  br label %bb679

bb679:
  br label %bb680

bb680:
  br label %bb681

bb681:
  br label %bb682

bb682:
  br label %bb683

bb683:
  br label %bb684

bb684:
  br label %bb685

bb685:
  br label %bb686

bb686:
  br label %bb687

bb687:
  br label %bb688

bb688:
  br label %bb689

bb689:
  br label %bb690

bb690:
  br label %bb691

bb691:
  br label %bb692

bb692:
  br label %bb693

bb693:
  br label %bb694

bb694:
  br label %bb695

bb695:
  br label %bb696

bb696:
  br label %bb697

bb697:
  br label %bb698

bb698:
  br label %bb699

bb699:
  br label %bb700

bb700:
  br label %bb701

bb701:
  br label %bb702

bb702:
  br label %bb703

bb703:
  br label %bb704

bb704:
  br label %bb705

bb705:
  br label %bb706

bb706:
  br label %bb707

bb707:
  br label %bb708

bb708:
  br label %bb709

bb709:
  br label %bb710

bb710:
  br label %bb711

bb711:
  br label %bb712

bb712:
  br label %bb713

bb713:
  br label %bb714

bb714:
  br label %bb715

bb715:
  br label %bb716

bb716:
  br label %bb717

bb717:
  br label %bb718

bb718:
  br label %bb719

bb719:
  br label %bb720

bb720:
  br label %bb721

bb721:
  br label %bb722

bb722:
  br label %bb723

bb723:
  br label %bb724

bb724:
  br label %bb725

bb725:
  br label %bb726

bb726:
  br label %bb727

bb727:
  br label %bb728

bb728:
  br label %bb729

bb729:
  br label %bb730

bb730:
  br label %bb731

bb731:
  br label %bb732

bb732:
  br label %bb733

bb733:
  br label %bb734

bb734:
  br label %bb735

bb735:
  br label %bb736

bb736:
  br label %bb737

bb737:
  br label %bb738

bb738:
  br label %bb739

bb739:
  br label %bb740

bb740:
  br label %bb741

bb741:
  br label %bb742

bb742:
  br label %bb743

bb743:
  br label %bb744

bb744:
  br label %bb745

bb745:
  br label %bb746

bb746:
  br label %bb747

bb747:
  br label %bb748

bb748:
  br label %bb749

bb749:
  br label %bb750

bb750:
  br label %bb751

bb751:
  br label %bb752

bb752:
  br label %bb753

bb753:
  br label %bb754

bb754:
  br label %bb755

bb755:
  br label %bb756

bb756:
  br label %bb757

bb757:
  br label %bb758

bb758:
  br label %bb759

bb759:
  br label %bb760

bb760:
  br label %bb761

bb761:
  br label %bb762

bb762:
  br label %bb763

bb763:
  br label %bb764

bb764:
  br label %bb765

bb765:
  br label %bb766

bb766:
  br label %bb767

bb767:
  br label %bb768

bb768:
  br label %bb769

bb769:
  br label %bb770

bb770:
  br label %bb771

bb771:
  br label %bb772

bb772:
  br label %bb773

bb773:
  br label %bb774

bb774:
  br label %bb775

bb775:
  br label %bb776

bb776:
  br label %bb777

bb777:
  br label %bb778

bb778:
  br label %bb779

bb779:
  br label %bb780

bb780:
  br label %bb781

bb781:
  br label %bb782

bb782:
  br label %bb783

bb783:
  br label %bb784

bb784:
  br label %bb785

bb785:
  br label %bb786

bb786:
  br label %bb787

bb787:
  br label %bb788

bb788:
  br label %bb789

bb789:
  br label %bb790

bb790:
  br label %bb791

bb791:
  br label %bb792

bb792:
  br label %bb793

bb793:
  br label %bb794

bb794:
  br label %bb795

bb795:
  br label %bb796

bb796:
  br label %bb797

bb797:
  br label %bb798

bb798:
  br label %bb799

bb799:
  br label %bb800

bb800:
  br label %bb801

bb801:
  br label %bb802

bb802:
  br label %bb803

bb803:
  br label %bb804

bb804:
  br label %bb805

bb805:
  br label %bb806

bb806:
  br label %bb807

bb807:
  br label %bb808

bb808:
  br label %bb809

bb809:
  br label %bb810

bb810:
  br label %bb811

bb811:
  br label %bb812

bb812:
  br label %bb813

bb813:
  br label %bb814

bb814:
  br label %bb815

bb815:
  br label %bb816

bb816:
  br label %bb817

bb817:
  br label %bb818

bb818:
  br label %bb819

bb819:
  br label %bb820

bb820:
  br label %bb821

bb821:
  br label %bb822

bb822:
  br label %bb823

bb823:
  br label %bb824

bb824:
  br label %bb825

bb825:
  br label %bb826

bb826:
  br label %bb827

bb827:
  br label %bb828

bb828:
  br label %bb829

bb829:
  br label %bb830

bb830:
  br label %bb831

bb831:
  br label %bb832

bb832:
  br label %bb833

bb833:
  br label %bb834

bb834:
  br label %bb835

bb835:
  br label %bb836

bb836:
  br label %bb837

bb837:
  br label %bb838

bb838:
  br label %bb839

bb839:
  br label %bb840

bb840:
  br label %bb841

bb841:
  br label %bb842

bb842:
  br label %bb843

bb843:
  br label %bb844

bb844:
  br label %bb845

bb845:
  br label %bb846

bb846:
  br label %bb847

bb847:
  br label %bb848

bb848:
  br label %bb849

bb849:
  br label %bb850

bb850:
  br label %bb851

bb851:
  br label %bb852

bb852:
  br label %bb853

bb853:
  br label %bb854

bb854:
  br label %bb855

bb855:
  br label %bb856

bb856:
  br label %bb857

bb857:
  br label %bb858

bb858:
  br label %bb859

bb859:
  br label %bb860

bb860:
  br label %bb861

bb861:
  br label %bb862

bb862:
  br label %bb863

bb863:
  br label %bb864

bb864:
  br label %bb865

bb865:
  br label %bb866

bb866:
  br label %bb867

bb867:
  br label %bb868

bb868:
  br label %bb869

bb869:
  br label %bb870

bb870:
  br label %bb871

bb871:
  br label %bb872

bb872:
  br label %bb873

bb873:
  br label %bb874

bb874:
  br label %bb875

bb875:
  br label %bb876

bb876:
  br label %bb877

bb877:
  br label %bb878

bb878:
  br label %bb879

bb879:
  br label %bb880

bb880:
  br label %bb881

bb881:
  br label %bb882

bb882:
  br label %bb883

bb883:
  br label %bb884

bb884:
  br label %bb885

bb885:
  br label %bb886

bb886:
  br label %bb887

bb887:
  br label %bb888

bb888:
  br label %bb889

bb889:
  br label %bb890

bb890:
  br label %bb891

bb891:
  br label %bb892

bb892:
  br label %bb893

bb893:
  br label %bb894

bb894:
  br label %bb895

bb895:
  br label %bb896

bb896:
  br label %bb897

bb897:
  br label %bb898

bb898:
  br label %bb899

bb899:
  br label %bb900

bb900:
  br label %bb901

bb901:
  br label %bb902

bb902:
  br label %bb903

bb903:
  br label %bb904

bb904:
  br label %bb905

bb905:
  br label %bb906

bb906:
  br label %bb907

bb907:
  br label %bb908

bb908:
  br label %bb909

bb909:
  br label %bb910

bb910:
  br label %bb911

bb911:
  br label %bb912

bb912:
  br label %bb913

bb913:
  br label %bb914

bb914:
  br label %bb915

bb915:
  br label %bb916

bb916:
  br label %bb917

bb917:
  br label %bb918

bb918:
  br label %bb919

bb919:
  br label %bb920

bb920:
  br label %bb921

bb921:
  br label %bb922

bb922:
  br label %bb923

bb923:
  br label %bb924

bb924:
  br label %bb925

bb925:
  br label %bb926

bb926:
  br label %bb927

bb927:
  br label %bb928

bb928:
  br label %bb929

bb929:
  br label %bb930

bb930:
  br label %bb931

bb931:
  br label %bb932

bb932:
  br label %bb933

bb933:
  br label %bb934

bb934:
  br label %bb935

bb935:
  br label %bb936

bb936:
  br label %bb937

bb937:
  br label %bb938

bb938:
  br label %bb939

bb939:
  br label %bb940

bb940:
  br label %bb941

bb941:
  br label %bb942

bb942:
  br label %bb943

bb943:
  br label %bb944

bb944:
  br label %bb945

bb945:
  br label %bb946

bb946:
  br label %bb947

bb947:
  br label %bb948

bb948:
  br label %bb949

bb949:
  br label %bb950

bb950:
  br label %bb951

bb951:
  br label %bb952

bb952:
  br label %bb953

bb953:
  br label %bb954

bb954:
  br label %bb955

bb955:
  br label %bb956

bb956:
  br label %bb957

bb957:
  br label %bb958

bb958:
  br label %bb959

bb959:
  br label %bb960

bb960:
  br label %bb961

bb961:
  br label %bb962

bb962:
  br label %bb963

bb963:
  br label %bb964

bb964:
  br label %bb965

bb965:
  br label %bb966

bb966:
  br label %bb967

bb967:
  br label %bb968

bb968:
  br label %bb969

bb969:
  br label %bb970

bb970:
  br label %bb971

bb971:
  br label %bb972

bb972:
  br label %bb973

bb973:
  br label %bb974

bb974:
  br label %bb975

bb975:
  br label %bb976

bb976:
  br label %bb977

bb977:
  br label %bb978

bb978:
  br label %bb979

bb979:
  br label %bb980

bb980:
  br label %bb981

bb981:
  br label %bb982

bb982:
  br label %bb983

bb983:
  br label %bb984

bb984:
  br label %bb985

bb985:
  br label %bb986

bb986:
  br label %bb987

bb987:
  br label %bb988

bb988:
  br label %bb989

bb989:
  br label %bb990

bb990:
  br label %bb991

bb991:
  br label %bb992

bb992:
  br label %bb993

bb993:
  br label %bb994

bb994:
  br label %bb995

bb995:
  br label %bb996

bb996:
  br label %bb997

bb997:
  br label %bb998

bb998:
  br label %bb999

bb999:
  br label %bb1000

bb1000:
  ; CHECK: call{{.*}}@dfsan_union
  ; CHECK-NOT: phi
  %ab = mul i32 %a, %b
  ret i32 %ab
}
