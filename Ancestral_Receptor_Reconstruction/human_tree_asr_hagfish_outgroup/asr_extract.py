# Extract ancestral state sequence

# Usage: python asr_extract.py iqtree.state Node#
# Example: python asr_extract.py proteinA.iqtree.state Node27

import numpy as np
import sys

def pp(l):
    l = l.split('\t')
    n = 3
    for aa in list(pp_d.keys()):
        try:
            pp_d[aa] = float(l[n].strip())
            n += 1
        except:
            pass
    return pp_d    

# load files
statefile = open(sys.argv[1],'r').readlines()
node = sys.argv[2]

# make pp dictionary
states = statefile[8].split('\t')[3:]
states = [state.replace('p_','').strip() for state in states]
pp_d = {}
for state in states:
    pp_d[state] = 0

# extract sequence and statistics
sequence = ''
posterior_probs = []

# record site info
site = []
prob = []

for line in statefile:
    if line.split('\t')[0] == node:
        sequence = sequence + line.split('\t')[2].strip()
        d = pp(line)
        if line.split('\t')[2] != '-':
            site.append(int(line.split('\t')[1].strip()))
            prob.append(float(d[line.split('\t')[2].strip()]))
        try:
            posterior_probs.append(d[line.split('\t')[2].strip()])
        except:
            # eg. an amino acid wasn't selected
            pass

mean_pp = np.mean(posterior_probs)
sd_pp = np.std(posterior_probs)
median_pp = np.median(posterior_probs)
length = len(sequence) - sequence.count('-')

print(('\n>'+node+'_ASR\n'+sequence+'\n'))
print(('Protein length: ' + str(length)))
print(('Mean posterior probability = ' + str(round(mean_pp,3)) + ' (standard deviation = ' + str(round(sd_pp,3)) + ', median = ' + str(round(median_pp,3)) + ')\n'))

out = open(node+'_ASR.fasta','w')
out.write('>'+node+'_ASR\n'+sequence)
out.close()


# import matplotlib.pyplot as plt
# plt.scatter(site,prob)
# plt.show()

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 4))
# plt.plot(site, prob, marker='o', linestyle='-', color='teal', markersize=3)
# plt.axhline(y=0.95, color='red', linestyle='--', label='High confidence threshold (0.95)')
# plt.title(f"Posterior Probabilities of Inferred Amino Acids at {node}")
# plt.xlabel("Site (Filtered Alignment Position)")
# plt.ylabel("Posterior Probability")
# plt.ylim(0, 1.05)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.show()