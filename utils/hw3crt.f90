SUBROUTINE HW3CRTT(XS, XF, L, LBDCND, BDXS, BDXF, YS, YF, M, &
                   MBDCND, BDYS, BDYF, ZS, ZF, N, NBDCND, BDZS, BDZF, ELMBDA,  &
                   LDIMF, MDIMF, F, PERTRB, IERROR, W)
      IMPLICIT REAL*8(A-H,O-Z)
      
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER , INTENT(IN) :: L
      INTEGER  :: LBDCND
      INTEGER , INTENT(IN) :: M
      INTEGER  :: MBDCND
      INTEGER , INTENT(IN) :: N
      INTEGER , INTENT(IN) :: NBDCND
      INTEGER  :: LDIMF
      INTEGER  :: MDIMF
      INTEGER , INTENT(INOUT) :: IERROR
      REAL*8 , INTENT(IN) :: XS
      REAL*8 , INTENT(IN) :: XF
      REAL*8 , INTENT(IN) :: YS
      REAL*8 , INTENT(IN) :: YF
      REAL*8 , INTENT(IN) :: ZS
      REAL*8 , INTENT(IN) :: ZF
      REAL*8 , INTENT(IN) :: ELMBDA
      REAL*8 , INTENT(IN) :: BDXS(MDIMF,*)
      REAL*8 , INTENT(IN) :: BDXF(MDIMF,*)
      REAL*8 , INTENT(IN) :: BDYS(LDIMF,*)
      REAL*8 , INTENT(IN) :: BDYF(LDIMF,*)
      REAL*8 , INTENT(IN) :: BDZS(LDIMF,*)
      REAL*8 , INTENT(IN) :: BDZF(LDIMF,*)
      REAL*8 , INTENT(INOUT) :: PERTRB
      REAL*8  :: F(LDIMF,MDIMF,*)
      REAL*8  :: W(*)
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------
      INTEGER :: MSTART, MSTOP, MP1, MP, MUNK, NP, NP1, NSTART, NSTOP, &
                 NUNK, LP1, LP, LSTART, LSTOP, J, K, LUNK, I, IWB, IWC, IWW, &
                 MSTPM1, LSTPM1, NSTPM1, NPEROD, IR
      REAL*8  :: DY,TWBYDY,C2,DZ,TWBYDZ,C3,DX,C1,TWBYDX,XLP,YLP,ZLP,S1,S2,S
 
      IERROR = 0
      IF (XF <= XS)               IERROR = 1
      IF (L < 5)                  IERROR = 2
      IF (LBDCND<0 .OR. LBDCND>4) IERROR = 3
      IF (YF <= YS)               IERROR = 4
      IF (M < 5)                  IERROR = 5
      IF (MBDCND<0 .OR. MBDCND>4) IERROR = 6
      IF (ZF <= ZS)               IERROR = 7
      IF (N < 5)                  IERROR = 8
      IF (NBDCND<0 .OR. NBDCND>4) IERROR = 9
      IF (LDIMF < L + 1)          IERROR = 10
      IF (MDIMF < M + 1)          IERROR = 11
      
      IF (IERROR > 0) RETURN 

      DY     = (YF - YS)/FLOAT(M)
      TWBYDY = 2.D0/DY
      C2     = 1.D0/DY**2
      MSTART = 1
      MSTOP  = M
      MP1    = M + 1
      MP     = MBDCND + 1
      
      GO TO (104,101,101,102,102) MP
  101 CONTINUE
      MSTART = 2
  102 CONTINUE
      GO TO (104,104,103,103,104) MP
  103 CONTINUE
      MSTOP = MP1
  104 CONTINUE
  
      MUNK   = MSTOP - MSTART + 1
      DZ     = (ZF - ZS)/FLOAT(N)
      TWBYDZ = 2.D0/DZ
      NP     = NBDCND + 1
      C3     = 1.D0/DZ**2
      NP1    = N + 1
      NSTART = 1
      NSTOP  = N
      
      GO TO (108,105,105,106,106) NP
  105 CONTINUE
      NSTART = 2
  106 CONTINUE
      GO TO (108,108,107,107,108) NP
  107 CONTINUE
      NSTOP = NP1
  108 CONTINUE
  
      NUNK   = NSTOP - NSTART + 1
      LP1    = L + 1
      DX     = (XF - XS)/FLOAT(L)
      C1     = 1.D0/DX**2
      TWBYDX = 2.D0/DX
      LP     = LBDCND + 1
      LSTART = 1
      LSTOP  = L
!
!     ENTER BOUNDARY DATA FOR X-BOUNDARIES.
!
      GO TO (122,109,109,112,112) LP
  109 CONTINUE
      LSTART = 2
      F(2,MSTART:MSTOP,NSTART:NSTOP) = F(2,MSTART:MSTOP,NSTART:NSTOP) - & 
                                       C1*F(1,MSTART:MSTOP,NSTART:NSTOP)
      GO TO 115
  112 CONTINUE
      F(1,MSTART:MSTOP,NSTART:NSTOP) = F(1,MSTART:MSTOP,NSTART:NSTOP) + &
                                       TWBYDX*BDXS(MSTART:MSTOP,NSTART:NSTOP)
  115 CONTINUE
      GO TO (122,116,119,119,116) LP
  116 CONTINUE
      F(L,MSTART:MSTOP,NSTART:NSTOP) = F(L,MSTART:MSTOP,NSTART:NSTOP) - &
                                       C1*F(LP1,MSTART:MSTOP,NSTART:NSTOP)
      GO TO 122
  119 CONTINUE
      LSTOP = LP1
      F(LP1,MSTART:MSTOP,NSTART:NSTOP) = F(LP1,MSTART:MSTOP,NSTART:NSTOP) - &
                                         TWBYDX*BDXF(MSTART:MSTOP,NSTART:NSTOP)
  122 CONTINUE
      LUNK = LSTOP - LSTART + 1
!
!     ENTER BOUNDARY DATA FOR Y-BOUNDARIES.
!
      GO TO (136,123,123,126,126) MP
  123 CONTINUE
      F(LSTART:LSTOP,2,NSTART:NSTOP) = F(LSTART:LSTOP,2,NSTART:NSTOP) - &
                                       C2*F(LSTART:LSTOP,1,NSTART:NSTOP)
      GO TO 129
  126 CONTINUE
      F(LSTART:LSTOP,1,NSTART:NSTOP) = F(LSTART:LSTOP,1,NSTART:NSTOP) + &
                                       TWBYDY*BDYS(LSTART:LSTOP,NSTART:NSTOP)
  129 CONTINUE
      GO TO (136,130,133,133,130) MP
  130 CONTINUE
      F(LSTART:LSTOP,M,NSTART:NSTOP) = F(LSTART:LSTOP,M,NSTART:NSTOP) - & 
                                       C2*F(LSTART:LSTOP,MP1,NSTART:NSTOP)
      GO TO 136
  133 CONTINUE
      F(LSTART:LSTOP,MP1,NSTART:NSTOP) = F(LSTART:LSTOP,MP1,NSTART:NSTOP) - &
                                         TWBYDY*BDYF(LSTART:LSTOP,NSTART:NSTOP)
  136 CONTINUE
      GO TO (150,137,137,140,140) NP
  137 CONTINUE
      F(LSTART:LSTOP,MSTART:MSTOP,2) = F(LSTART:LSTOP,MSTART:MSTOP,2) - &
                                       C3*F(LSTART:LSTOP,MSTART:MSTOP,1)
      GO TO 143
  140 CONTINUE
      F(LSTART:LSTOP,MSTART:MSTOP,1) = F(LSTART:LSTOP,MSTART:MSTOP,1) + &
                                       TWBYDZ*BDZS(LSTART:LSTOP,MSTART:MSTOP)
  143 CONTINUE
      GO TO (150,144,147,147,144) NP
  144 CONTINUE
      F(LSTART:LSTOP,MSTART:MSTOP,N) = F(LSTART:LSTOP,MSTART:MSTOP,N) - &
                                       C3*F(LSTART:LSTOP,MSTART:MSTOP,NP1)
      GO TO 150
  147 CONTINUE
      F(LSTART:LSTOP,MSTART:MSTOP,NP1) = F(LSTART:LSTOP,MSTART:MSTOP,NP1) - &
                                         TWBYDZ*BDZF(LSTART:LSTOP,MSTART:MSTOP)
     
!
!     DEFINE A,B,C COEFFICIENTS IN W-ARRAY.
!

  150 CONTINUE
      IWB = NUNK + 1
      IWC = IWB + NUNK
      IWW = IWC + NUNK
      W(:NUNK) = C3
      W(IWC:NUNK-1+IWC) = C3
      W(IWB:NUNK-1+IWB) = (-2.D0*C3) + ELMBDA
      GO TO (155,155,153,152,152) NP
  152 CONTINUE
      W(IWC) = 2.D0*C3
  153 CONTINUE
      GO TO (155,155,154,154,155) NP
  154 CONTINUE
      W(IWB-1) = 2.D0*C3
  155 CONTINUE
      PERTRB = 0.D0
!
!     FOR SINGULAR PROBLEMS ADJUST DATA TO INSURE A SOLUTION WILL EXIST.
!
      GO TO (156,172,172,156,172) LP
  156 CONTINUE
      GO TO (157,172,172,157,172) MP
  157 CONTINUE
      GO TO (158,172,172,158,172) NP
  158 CONTINUE
      IF (ELMBDA >= 0.D0) THEN
         IF (ELMBDA /= 0.D0) THEN
            IERROR = 12
         ELSE
            MSTPM1 = MSTOP - 1
            LSTPM1 = LSTOP - 1
            NSTPM1 = NSTOP - 1
            XLP = (2 + LP)/3
            YLP = (2 + MP)/3
            ZLP = (2 + NP)/3
            S1 = 0.D0
            DO K = 2, NSTPM1
               DO J = 2, MSTPM1
                  S1 = S1 + SUM(F(2:LSTPM1,J,K))
                  S1 = S1 + (F(1,J,K)+F(LSTOP,J,K))/XLP
               END DO
               S2 = SUM(F(2:LSTPM1,1,K)+F(2:LSTPM1,MSTOP,K))
               S2 = (S2 + (F(1,1,K) + F(1,MSTOP,K) + F(LSTOP,1,K) + &
                           F(LSTOP,MSTOP,K))/XLP)/YLP
               S1 = S1 + S2
            END DO
            S = (F(1,1,1) + F(LSTOP,1,1) + F(1,1,NSTOP) + F(LSTOP,1,NSTOP) + &
                 F(1,MSTOP,1) + F(LSTOP,MSTOP,1) + F(1,MSTOP,NSTOP) + &
                 F(LSTOP,MSTOP,NSTOP))/(XLP*YLP)
            DO J = 2, MSTPM1
               S = S + SUM(F(2:LSTPM1,J,1)+F(2:LSTPM1,J,NSTOP))
            END DO
            S2 = 0.D0
            S2 = SUM(F(2:LSTPM1,1,1) + F(2:LSTPM1,1,NSTOP) + F(2:LSTPM1,MSTOP,1) + &
                     F(2:LSTPM1,MSTOP,NSTOP))
            S = S2/YLP + S
            S2 = 0.D0
            S2 = SUM(F(1,2:MSTPM1,1) + F(1,2:MSTPM1,NSTOP) + F(LSTOP,2:MSTPM1,1) + &
                     F(LSTOP,2:MSTPM1,NSTOP))
            S = S2/XLP + S
            PERTRB = (S/ZLP + S1)/((FLOAT(LUNK + 1) - XLP)*(FLOAT(MUNK + 1) &
                     - YLP)*(FLOAT(NUNK + 1) - ZLP))
            F(:LUNK,:MUNK,:NUNK) = F(:LUNK,:MUNK,:NUNK) - PERTRB
         ENDIF
      ENDIF
  172 CONTINUE
      NPEROD = 0
      IF (NBDCND /= 0) THEN
         NPEROD = 1
         W(1) = 0.D0
         W(IWW-1) = 0.D0
      ENDIF
      CALL POIS3DD (LBDCND, LUNK, C1, MBDCND, MUNK, C2, NPEROD, NUNK, W, &
                    W(IWB), W(IWC), LDIMF, MDIMF, F(LSTART,MSTART,NSTART), &
                    IR, W(IWW))
!
!     FILL IN SIDES FOR PERIODIC BOUNDARY CONDITIONS.
!
      IF (LP == 1) THEN
         IF (MP == 1) THEN
            F(1,MP1,NSTART:NSTOP) = F(1,1,NSTART:NSTOP)
            MSTOP = MP1
         ENDIF
         IF (NP == 1) THEN
            F(1,MSTART:MSTOP,NP1) = F(1,MSTART:MSTOP,1)
            NSTOP = NP1
         ENDIF
         F(LP1,MSTART:MSTOP,NSTART:NSTOP) = F(1,MSTART:MSTOP,NSTART:NSTOP)
      ENDIF
      IF (MP == 1) THEN
         IF (NP == 1) THEN
            F(LSTART:LSTOP,1,NP1) = F(LSTART:LSTOP,1,1)
            NSTOP = NP1
         ENDIF
         F(LSTART:LSTOP,MP1,NSTART:NSTOP) = F(LSTART:LSTOP,1,NSTART:NSTOP)
      ENDIF
      IF (NP == 1) THEN
         F(LSTART:LSTOP,MSTART:MSTOP,NP1) = F(LSTART:LSTOP,MSTART:MSTOP,1)
      ENDIF
      
      RETURN 
!
! REVISION HISTORY---
!
! SEPTEMBER 1973    VERSION 1
! APRIL     1976    VERSION 2
! JANUARY   1978    VERSION 3
! DECEMBER  1979    VERSION 3.1
! FEBRUARY  1985    DOCUMENTATION UPGRADE
! NOVEMBER  1988    VERSION 3.2, FORTRAN 77 CHANGES
! June      2004    Version 5.0, Fortran 90 changes
!-----------------------------------------------------------------------

END SUBROUTINE HW3CRTT
