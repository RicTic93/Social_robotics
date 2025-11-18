config = {
    'num_objects': 3,    # Number d'objets dans la scène

    'dynamic_humans_count': 3,     # Nombre d'humains dynamic
    'static_groups':[              # Parametre de static_groups
        {
            'count': 3,
            'formation': 'converging',
            'center_pos': [3.0, 7.0],
            'radius': 1.0,
        },
        {
            'count': 2,
            'formation': 'diverging',
            'center_pos': [7.0, 3.0],
            'radius': 1.0,
        }
            ],
                
    'max_steps': 500,    # Nombre maximal de steps par épisode

    'social_distance': 1.5   # Distance entre les humans
}
