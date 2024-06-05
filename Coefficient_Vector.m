function cofi = Coefficient_Vector(dim,Iter,MaxIter)

    %Nonlinear control factor
    a2=-2+log2(1+(1-(Iter)/MaxIter).^2);
    u=randn(1,dim);
    v=randn(1,dim); 

    cofi(1,:)=rand(1,dim);
    cofi(2,:)= (a2+1)+rand;
    cofi(3,:)= a2.*randn(1,dim);
    cofi(4,:)= u.*v.^2.*cos((rand*2)*u);
            
end