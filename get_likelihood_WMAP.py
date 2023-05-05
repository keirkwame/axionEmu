def get_likelihood_WMAP(d_ell_dict_indiv):
  #print(d_ell_dict_indiv['zero_prior'])
  if not d_ell_dict_indiv['zero_prior']:
    likelihood_indiv, _ = d_ell_dict_indiv['likelihood_object'].loglike(d_ell_dict_indiv)
    return likelihood_indiv
  else:
    return -1.e+10
