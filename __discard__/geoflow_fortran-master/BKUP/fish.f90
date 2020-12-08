
      MODULE fish
	CONTAINs
	
	SUBROUTINE allocatfish(irwk,icwk,wsave,ierror)
	IMPLICIT REAL*8(A-H,O-Z)
	
	REAL*8, ALLOCATABLE, DIMENSION(:) :: wsave
	INTEGER, INTENT(IN) :: irwk,icwk
	INTEGER, INTENT(INOUT) :: ierror
	INTEGER :: istatus

	DEALLOCATE(wsave,STAT = istatus)
	if (irwk > 0) then
	     ALLOCATE(wsave(irwk),STAT = istatus)
	end if
	ierror = 0
	
	if (istatus .ne. 0 ) then
	  ierror = 20
	END IF
	
	RETURN
	END SUBROUTINE allocatfish

	SUBROUTINE BLK_space(N,M,irwk,icwk)
	IMPLICIT REAL*8(A-H,O-Z)
	
	INTEGER,INTENT(IN) :: N,M
	INTEGER,INTENT(OUT) :: irwk,icwk
	INTEGER :: L,log2n

	log2n = 1
	do
	   log2n = log2n+1
	   if (n+1 <= 2**log2n) EXIT
	end do
	
	L = 2**(log2n+1)
	irwk = (log2n-2)*L+5+MAX0(2*N,6*M)+log2n+2*n
	icwk = ((log2n-2)*L+5+log2n)/2+3*M+N
	
	RETURN
	END SUBROUTINE BLK_space

	SUBROUTINE GEN_space(N,M,irwk)
	IMPLICIT REAL*8(A-H,O-Z)
	
	INTEGER,INTENT(IN) :: N,M
	INTEGER,INTENT(OUT) :: irwk
	INTEGER :: log2n
	
	log2n = 1
	do
	   log2n = log2n+1
	   if (n+1 <= 2**log2n) EXIT
	end do
	irwk = 4*N + (10 + log2n)*M
	RETURN
	END SUBROUTINE GEN_space

	SUBROUTINE fishfin(wsave)
	IMPLICIT REAL*8(A-H,O-Z)
	
	REAL*8, ALLOCATABLE, DIMENSION(:) :: wsave
	INTEGER :: istatus
	DEALLOCATE(wsave,STAT = istatus)
	RETURN
	END SUBROUTINE fishfin

      END MODULE fish
