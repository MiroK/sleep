    # Nistche method for the noflow BC 
    alpha = Constant(1./10)
    beta = Constant(10)
    h = CellSize(mesh)

    for tag,value in noflow_bcs:   
       system +=- dot(n,t(u,p))*dot(v,n)*ds(tag) - dot(u,n)*dot(n,t(v,q))*ds(tag) \
       system += beta/h*dot(u,n)*dot(v,n)*ds(tag) \
       system += alpha*h**2*dot(f - grad(p), grad(q))*dx