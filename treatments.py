"""
Treatment recommendations for mango leaf diseases
"""

DISEASE_TREATMENTS = {
    'Anthracnose': {
        'disease_info': 'A fungal disease that causes dark, sunken lesions on leaves and fruits.',
        'symptoms': [
            'Dark, circular spots on leaves',
            'Sunken lesions on fruits',
            'Leaf yellowing and drop',
            'Twig dieback'
        ],
        'treatment': {
            'immediate': [
                'Remove and destroy infected plant parts',
                'Improve air circulation around trees',
                'Avoid overhead watering'
            ],
            'chemical': [
                'Apply copper-based fungicide (Copper oxychloride 50% WP @ 3g/L)',
                'Spray Mancozeb 75% WP @ 2.5g/L',
                'Use Propiconazole 25% EC @ 1ml/L during dry weather'
            ],
            'organic': [
                'Neem oil spray (5ml/L water)',
                'Baking soda solution (5g/L water)',
                'Compost tea application'
            ],
            'preventive': [
                'Maintain proper plant spacing',
                'Regular pruning for air circulation',
                'Balanced fertilization',
                'Avoid water stress'
            ]
        },
        'severity': 'Moderate to High'
    },
    
    'Bacterial_Canker': {
        'disease_info': 'A bacterial disease causing cankers on stems and branches.',
        'symptoms': [
            'Dark, sunken cankers on branches',
            'Yellowing and wilting of leaves',
            'Gum exudation from infected areas',
            'Branch dieback'
        ],
        'treatment': {
            'immediate': [
                'Prune and destroy infected branches 15cm below visible symptoms',
                'Disinfect pruning tools with 70% alcohol',
                'Remove water sprouts and suckers'
            ],
            'chemical': [
                'Copper hydroxide spray (2-3g/L)',
                'Streptomycin sulfate (100ppm)',
                'Bactericide application during cool, dry weather'
            ],
            'organic': [
                'Bordeaux mixture (1% solution)',
                'Garlic and chili extract spray',
                'Biocontrol agents (Bacillus subtilis)'
            ],
            'preventive': [
                'Avoid mechanical injuries',
                'Control insect vectors',
                'Proper irrigation management',
                'Regular monitoring and early detection'
            ]
        },
        'severity': 'High'
    },
    
    'Cutting_Weevil': {
        'disease_info': 'Insect pest that damages young shoots and growing tips.',
        'symptoms': [
            'Wilting and drying of young shoots',
            'Small holes in stems',
            'Presence of adult weevils',
            'Stunted growth'
        ],
        'treatment': {
            'immediate': [
                'Remove and destroy infested shoots',
                'Collect and destroy adult weevils manually',
                'Clean cultivation around trees'
            ],
            'chemical': [
                'Chlorpyrifos 20% EC @ 2ml/L',
                'Imidacloprid 17.8% SL @ 0.5ml/L',
                'Spray insecticides during early morning or evening'
            ],
            'organic': [
                'Neem-based insecticides (1500-2000ppm)',
                'Pheromone traps for monitoring',
                'Encourage natural predators'
            ],
            'preventive': [
                'Regular monitoring of shoots',
                'Proper pruning practices',
                'Avoid water stress',
                'Maintain orchard hygiene'
            ]
        },
        'severity': 'Moderate'
    },
    
    'Die_Back': {
        'disease_info': 'Progressive death of shoots, branches, and twigs from tips downward.',
        'symptoms': [
            'Yellowing and browning of leaves',
            'Progressive death from tip to base',
            'Bark cracking and peeling',
            'Reduced fruit production'
        ],
        'treatment': {
            'immediate': [
                'Prune affected branches 30cm below visible symptoms',
                'Burn or bury pruned material',
                'Improve drainage if waterlogged'
            ],
            'chemical': [
                'Copper fungicide spray (3g/L)',
                'Carbendazim 50% WP @ 1g/L',
                'Apply during dry weather conditions'
            ],
            'organic': [
                'Compost application to improve soil health',
                'Mulching around tree base',
                'Biocontrol fungi (Trichoderma spp.)'
            ],
            'preventive': [
                'Balanced nutrition management',
                'Proper water management',
                'Regular health monitoring',
                'Stress reduction measures'
            ]
        },
        'severity': 'High'
    },
    
    'Gall_Midge': {
        'disease_info': 'Small flies that cause gall formation on leaves and shoots.',
        'symptoms': [
            'Small galls on leaf surfaces',
            'Distorted and curled leaves',
            'Reduced photosynthesis',
            'Stunted growth'
        ],
        'treatment': {
            'immediate': [
                'Remove and destroy galls',
                'Prune affected shoots',
                'Monitor for adult emergence'
            ],
            'chemical': [
                'Dimethoate 30% EC @ 2ml/L',
                'Malathion 50% EC @ 2ml/L',
                'Apply when adult emergence is noticed'
            ],
            'organic': [
                'Yellow sticky traps for adults',
                'Neem oil spray (5ml/L)',
                'Encourage parasitic wasps'
            ],
            'preventive': [
                'Regular inspection of new growth',
                'Maintain orchard cleanliness',
                'Avoid excessive nitrogen fertilization',
                'Proper irrigation scheduling'
            ]
        },
        'severity': 'Moderate'
    },
    
    'Healthy': {
        'disease_info': 'The leaf appears healthy with no visible disease symptoms.',
        'symptoms': [
            'Green, vigorous foliage',
            'No spots or lesions',
            'Normal growth pattern',
            'Good photosynthetic activity'
        ],
        'treatment': {
            'immediate': [
                'Continue current management practices',
                'Monitor regularly for early disease detection'
            ],
            'chemical': [
                'No chemical treatment needed',
                'Maintain preventive spray schedule if applicable'
            ],
            'organic': [
                'Continue organic soil amendments',
                'Maintain beneficial insect populations'
            ],
            'preventive': [
                'Regular monitoring',
                'Balanced nutrition',
                'Proper irrigation',
                'Good orchard hygiene',
                'Preventive fungicide applications during susceptible periods'
            ]
        },
        'severity': 'None'
    },
    
    'Powdery_Mildew': {
        'disease_info': 'Fungal disease causing white powdery growth on leaf surfaces.',
        'symptoms': [
            'White powdery coating on leaves',
            'Yellowing and curling of leaves',
            'Reduced photosynthesis',
            'Premature leaf drop'
        ],
        'treatment': {
            'immediate': [
                'Remove severely affected leaves',
                'Improve air circulation',
                'Reduce humidity around plants'
            ],
            'chemical': [
                'Sulfur-based fungicides (3g/L)',
                'Triadimefon 25% WP @ 1g/L',
                'Myclobutanil 10% WP @ 1g/L'
            ],
            'organic': [
                'Milk spray (1:10 dilution with water)',
                'Baking soda solution (5g/L)',
                'Potassium bicarbonate spray'
            ],
            'preventive': [
                'Plant in well-ventilated areas',
                'Avoid overhead irrigation',
                'Regular monitoring',
                'Balanced fertilization (avoid excess nitrogen)'
            ]
        },
        'severity': 'Moderate'
    },
    
    'Sooty_Mould': {
        'disease_info': 'Black fungal growth on honeydew secreted by sucking insects.',
        'symptoms': [
            'Black sooty deposits on leaves',
            'Reduced photosynthesis',
            'Yellowing of leaves',
            'Presence of aphids or scale insects'
        ],
        'treatment': {
            'immediate': [
                'Wash off sooty deposits with water',
                'Control underlying insect infestation',
                'Improve air circulation'
            ],
            'chemical': [
                'Control sucking insects with Imidacloprid 17.8% SL @ 0.5ml/L',
                'Dimethoate 30% EC @ 2ml/L for aphid control',
                'Copper fungicide if needed'
            ],
            'organic': [
                'Insecticidal soap spray',
                'Neem oil application (5ml/L)',
                'Encourage beneficial insects'
            ],
            'preventive': [
                'Regular monitoring for sucking insects',
                'Ant control (ants protect aphids)',
                'Maintain orchard cleanliness',
                'Balanced nutrition management'
            ]
        },
        'severity': 'Low to Moderate'
    }
}

def get_treatment_recommendation(disease_class):
    """Get comprehensive treatment recommendation for a disease"""
    if disease_class in DISEASE_TREATMENTS:
        return DISEASE_TREATMENTS[disease_class]
    else:
        return {
            'disease_info': 'Disease information not available.',
            'symptoms': ['Unknown symptoms'],
            'treatment': {
                'immediate': ['Consult with agricultural extension officer'],
                'chemical': ['Seek professional advice'],
                'organic': ['Apply general organic practices'],
                'preventive': ['Maintain good agricultural practices']
            },
            'severity': 'Unknown'
        }

def get_severity_color(severity):
    """Get color code for severity level"""
    severity_colors = {
        'None': 'success',
        'Low to Moderate': 'warning',
        'Moderate': 'warning', 
        'Moderate to High': 'danger',
        'High': 'danger',
        'Unknown': 'secondary'
    }
    return severity_colors.get(severity, 'secondary')
