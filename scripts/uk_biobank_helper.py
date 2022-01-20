#-------------------------------------------------------------------------------
# Functions specific to UK Biobank analysis
#-------------------------------------------------------------------------------
def find_threshold (beta, af_function, xvals) :
    """
    Takes an effect size (beta) and function relating effect size to minimum
    allele frequency (af_function) defined over some set of beta values (xvals).
    Returns the minimum allele frequency associated with each value of beta.

    Note that if the site can not be detected as significant, the function
    returns a value of one.
    """

    if beta < np.min (af_function (xvals)) :
        minaf = 1.0
    else :
        minaf = np.min (xnew[np.where (af_function (xvals) - beta < 0)])

    return minaf


def prune_snps (snpdict, distance) :
    """
    Prunes SNPs based on p-values to be a minimum of 'distance' apart from each
    other.
    Arguments: snpdict, a nested dictionary indexed by chromosome and position
                        and containing keys 'pval' and 'beta'
               distance, the minimum distance between a focal SNP and other SNPs
                         to each side
    Returns a list of the positions (chrom,pos) of the pruned SNPs and a list
            of the pruned effect sizes.
    """

    pruned_indices = list ()
    pruned_betas = list ()

    for chrom in snpdict.keys () :
        signif_pvalues = [snpdict[chrom][pos]['pval'] for pos in snpdict[chrom].keys()]
        signif_bp      = [pos for pos in snpdict[chrom].keys()]
        print('chrom: ' + str(chrom) + ', ' + str(signif_bp[:10]))

        while len (signif_bp) > 0 :
            max_index = np.where (signif_pvalues == np.min (signif_pvalues))[0][0]
            max_bp    = signif_bp[max_index]
            pruned_indices.append ((chrom, max_bp))
            pruned_betas.append (snpdict[chrom][max_bp]['beta'])

            # remove focal snp from all lists
            signif_pvalues.pop (max_index)
            signif_bp.pop (max_index)

            # right side
            nearest_right = True
            while nearest_right :
                if max_index <= (len(signif_bp) -1 ) :
                    right_distance = signif_bp[max_index] - max_bp

                    if right_distance < mindistance :
                        signif_bp.pop (max_index)
                        signif_pvalues.pop (max_index)
                    else :
                        nearest_right = False
                else :
                    nearest_right = False

            # left side
            nearest_left = True
            while nearest_left :
                if (max_index - 1) >= 0 :
                    left_distance = max_bp - signif_bp[max_index-1]

                    if left_distance < mindistance :
                        signif_bp.pop (max_index-1)
                        signif_pvalues.pop (max_index-1)
                        max_index -= 1 # increment index
                    else :
                        nearest_left = False
                else :
                    nearest_left = False

    return pruned_indices, pruned_betas
