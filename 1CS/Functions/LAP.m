function [out]=LAP(in)%?????hologram???512?512double?????
[Nx Ny]=size(in);
k=0;
for ii=2:(Nx-1)
    for jj=2:(Ny-1)
        k=k+(in(ii+1,jj)+in(ii-1,jj)+in(ii,jj+1)+in(ii,jj-1)-4*in(ii,jj))^2;
    end
end
out=k;