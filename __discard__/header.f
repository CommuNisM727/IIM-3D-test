      SUBROUTINE HW3CRT (XS,XF,L,LBDCND,BDXS,BDXF,YS,YF,M,MBDCND,BDYS,
     1                   BDYF,ZS,ZF,N,NBDCND,BDZS,BDZF,ELMBDA,LDIMF,
     2                   MDIMF,F,PERTRB,IERROR,W)

      REAL*8 XS, XF, BDXS, BDXF
      REAL*8 YS, YF, BDYS, BDYF
      REAL*8 ZS, ZF, BDZS, BDZF
      REAL*8 F, W
Cf2py intent(in) W
Cf2py intent(inout) F, IERROR, W

      DIMENSION       BDXS(MDIMF,*)          ,BDXF(MDIMF,*)          ,
     1                BDYS(LDIMF,*)          ,BDYF(LDIMF,*)          ,
     2                BDZS(LDIMF,*)          ,BDZF(LDIMF,*)          ,
     3                F(LDIMF,MDIMF,*)       ,W(*)

      IERROR = 0
      IF (XF .LE. XS) IERROR = 1
      IF (L .LT. 5) IERROR = 2
      IF (LBDCND.LT.0 .OR. LBDCND.GT.4) IERROR = 3
      IF (YF .LE. YS) IERROR = 4
      IF (M .LT. 5) IERROR = 5
      IF (MBDCND.LT.0 .OR. MBDCND.GT.4) IERROR = 6
      IF (ZF .LE. ZS) IERROR = 7
      IF (N .LT. 5) IERROR = 8
      IF (NBDCND.LT.0 .OR. NBDCND.GT.4) IERROR = 9
      IF (LDIMF .LT. L+1) IERROR = 10
      IF (MDIMF .LT. M+1) IERROR = 11

      

      print *, XS, XF
      print *, YS, YF
      print *, ZS, ZF

      print *, IEEROR  
      IF (IERROR .NE. 0) GO TO 188
      print *, IEEROR

      DY = (YF-YS)/FLOAT(M)
      TWBYDY = 2./DY
      C2 = 1./(DY**2)
      MSTART = 1
      MSTOP = M
      MP1 = M+1
      MP = MBDCND+1
      GO TO (104,101,101,102,102),MP
  101 MSTART = 2
  102 GO TO (104,104,103,103,104),MP
  103 MSTOP = MP1
  104 MUNK = MSTOP-MSTART+1
      DZ = (ZF-ZS)/FLOAT(N)
      TWBYDZ = 2./DZ
      NP = NBDCND+1
      C3 = 1./(DZ**2)
      NP1 = N+1
      NSTART = 1
      NSTOP = N
      GO TO (108,105,105,106,106),NP
  105 NSTART = 2
  106 GO TO (108,108,107,107,108),NP
  107 NSTOP = NP1
  108 NUNK = NSTOP-NSTART+1
      LP1 = L+1
      DX = (XF-XS)/FLOAT(L)
      C1 = 1./(DX**2)
      TWBYDX = 2./DX
      LP = LBDCND+1
      LSTART = 1
      LSTOP = L
 
      DO 201 I=1,4
        DO 200 J=1,4
            DO 199 K=1,4
                print *, I, J, K, F(I, J, K)
C                F(I, J, K) = 0
                
  199 CONTINUE
  200 CONTINUE
  201 CONTINUE
  188 CONTINUE
      RETURN
      END