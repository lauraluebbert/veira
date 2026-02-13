def find_keys_to_scan(data):
    keys_to_scan = []
    for key in data:
        if '___' in key:
            keys_to_scan.append(key.split('___')[0])
        else:
            keys_to_scan.append(key)
    return sorted(set(keys_to_scan))

def gather_data_specific_key(data, query_key):
    results = []
    for key in data:
        if query_key in key:
            if '___' in key:
                value = key.split('___')[1]
                if data[key] == 1:
                    results.append(value)
            elif query_key == key:
                results.append(data[key])
    return results

def generate_patient_data_vignette(patient_data, df_field_definitions):
    
    useful_mappings = {
        'Indicate your housing material': 'housing material',
        'feverd': 'fever duration',
        'headached': 'headache duration',
        'Indicate your roofing type': 'roofing type',
        'Is your housing located around any of the following:Check as applicable': 'house is located near',
        'Any known comorbidities and/or chronic infections?': 'has a comorbidity/known chronic infection of',
        'Which of the sources of water do you use?Check as applicable': 'water source used',
        'Appearance': 'appearance of urine',
        'Nitrite': 'nitrite in urine',
        'joint': 'joint pain',
        'abdominal': 'abdominal pain',
        'Back': 'back pain',
        'muscle': 'muscle pain',
    }

    patient_vignette = ''
    for i in find_keys_to_scan(patient_data):
        try:
            options = df_field_definitions[df_field_definitions["Field Name"] == i].get("Choices", None)
            field_label = df_field_definitions[df_field_definitions["Field Name"] == i].get("Field Label", None).item()
            if options.isna().item() is not True:
                options = options.item().split("|")
                options = {int(x.split(",")[0].strip()): x.split(",")[1].strip() for x in options}
                pat_dat = gather_data_specific_key(patient_data, i)
                for res in pat_dat:
                    if res is not None and int(res) in options:
                        field_label = useful_mappings.get(field_label, field_label)
                        if options[int(res)] == '':
                            patient_vignette += f'The patients has {field_label}.'
                        else:
                            patient_vignette += f'The patients {field_label} is {options[int(res)]}. '
            else:
                pat_dat = gather_data_specific_key(patient_data, i)
                if i == 'toilet' and 1 in pat_dat:
                    patient_vignette += 'The patients has used a public toilet. '
                elif i == 'weight':
                    patient_vignette += f'The patients weight is {pat_dat[0]}. '
                elif i == 'pulse_rate':
                    patient_vignette += f'The patients pulse rate is {pat_dat[0]}. '
                elif i == 'family_bed' and 1 in pat_dat:
                    patient_vignette += 'The patients was sleeping in a family bed. '
                elif i == 'bush meet' and 1 in pat_dat:
                    patient_vignette += 'The patients has recently eaten bush meat. '
                elif i == 'wild_contact' and 1 in pat_dat:
                    patient_vignette += 'The patients has had contact with wild animals. '
                elif i == 'unprotected_sex' and 1 in pat_dat:
                    patient_vignette += 'The patients has had unprotected sex. '
                elif i == 'domestic_contact' and 1 in pat_dat:
                    patient_vignette += 'The patients has had contact with a domestic animal. '
                elif i == 'rodent_touch' and 1 in pat_dat:
                    patient_vignette += 'The patients has touched a rodent. '
                elif i == 'sharp_objects' and 1 in pat_dat:
                    patient_vignette += 'The patients has had contact with sharp objects. '
                elif i == 'food_bowls' and 1 in pat_dat:
                    patient_vignette += 'The patients stores food in open bowls. '
                elif i == 'other_contacts' and 1 in pat_dat:
                    patient_vignette += 'The patients has had other contacts. '
                elif i == 'sbp':
                    patient_vignette += f'The patients systolic blood pressure is {pat_dat[0]}. '
                elif i == 'family_room' and 1 in pat_dat:
                    patient_vignette += 'The patients was sleeping in the same room as their family. '
                elif i == 'travel' and 1 in pat_dat:
                    patient_vignette += 'The patients has traveled recently. '
                elif i == 'funeral_body' and 1 in pat_dat:
                    patient_vignette += 'The patients has handled a dead body. '
                elif i == 'insecticides' and 1 in pat_dat:
                    patient_vignette += 'The patients uses insecticides. '
                elif i == 'clean_water' and 1 in pat_dat:
                    patient_vignette += 'The patients has access to clean water. '
                elif i == 'travel_contact' and 1 in pat_dat:
                    patient_vignette += 'The patient has been in a vehicle with a sick person. '
                elif i == 'blood_transfusion' and 1 in pat_dat:
                    patient_vignette += 'The patients has recently received a blood transfusion. '
                elif i == 'family_care' and 1 in pat_dat:
                    patient_vignette += 'The patients has cared for a family member. '
                elif i == 'uncooked_meat' and 1 in pat_dat:
                    patient_vignette += 'The patients has touched uncooked meat. '
                elif i == 'food_container' and 1 in pat_dat:
                    patient_vignette += 'The patients stores food in covered containers. '
                elif i == 'eat_rodent' and 1 in pat_dat:
                    patient_vignette += 'The patients has eaten rodents. '
                elif (i == 'mosquito_bite' and 1 in pat_dat) or (i == 'mosquitobite' and 1 in pat_dat):
                    patient_vignette += 'The patients has bite marks of mosquitos. '
                elif i == 'food_sacs' and 1 in pat_dat:
                    patient_vignette += 'The patients stores food in food sacs. '
                elif i == 'family_fluids' and 1 in pat_dat:
                    patient_vignette += 'The patients has been exposed to family fluids. '
                elif i == 'symptomhistory' and 1 in pat_dat:
                    patient_vignette += 'The patients has a history of similar symptoms. '
                elif i == 'temperature':
                    patient_vignette += f'The patients temperature is {pat_dat[0]}. '
                elif i == 'mosquitonet' and 1 in pat_dat:
                    patient_vignette += 'The patients uses a mosquito net. '
                elif i == 'dbp':
                    patient_vignette += f'The patients diastolic blood pressure is {pat_dat[0]}. '
                elif i == 'hc_ppe' and 1 in pat_dat:
                    patient_vignette += 'The patients uses personal protective equipment. '
                elif i == 'respiratory_rate':
                    patient_vignette += f'The patients respiratory rate is {pat_dat[0]}. '
                elif i == 'funeral' and 1 in pat_dat:
                    patient_vignette += 'The patients has attended a funeral. '
                elif i == 'family_sick' and 1 in pat_dat:
                    patient_vignette += 'The patient has had recent sick family members. '
                elif i == 'contact_fluids' and 1 in pat_dat:
                    patient_vignette += 'The patient has been in contact with blood or body fluids. '
                elif i == 'haircut' and 1 in pat_dat:
                    patient_vignette += 'The patient has had a recent haircut with a public or shared clipper. '
                elif i == 'hc' and 1 in pat_dat:
                    patient_vignette += 'The patients has attended to a sick person as a healthcare worker. '
                elif i == 'surgery' and 1 in pat_dat:
                    patient_vignette += 'The patients has undergone surgery. '
                elif i[-1] == 'c' and i != 'hc' and None not in pat_dat:
                    if len(pat_dat) > 1:
                        print(f"Expected only one value for '{i}', but got multiple.")
                        import pdb; pdb.set_trace()
                    if len(pat_dat[0]) > 1 and i == 'domestic_contactc':
                        patient_vignette += f'The patients has had contact with domestic animals: {", ".join(pat_dat[0])}. '
                    elif len(pat_dat[0]) > 1 and i == 'othersymptomsc':
                        patient_vignette += f'The patients has symptoms including: {", ".join(pat_dat[0])}. '
                    elif len(pat_dat[0]) > 1 and i == 'feverc':
                        patient_vignette += f'The patients fever has the following properties: {", ".join(pat_dat[0])}. '
                    elif len(pat_dat[0]) > 1 and i == 'jointc':
                        patient_vignette += f'The patient has joint pain in the following areas: {", ".join(pat_dat[0])}. '
                    elif len(pat_dat[0]) > 1 and i == 'headachec':
                        patient_vignette += f'The patient has headache with the following properties: {", ".join(pat_dat[0])}. '
                    elif len(pat_dat[0]) > 1 and i == 'musclec':
                        patient_vignette += f'The patient has muscle pain in the following areas: {", ".join(pat_dat[0])}. '
                    elif len(pat_dat[0]) > 1 and i == 'vomitc':
                        patient_vignette += f'The patient has vomited with the following properties: {", ".join(pat_dat[0])}. '
                    elif len(pat_dat[0]) > 1 and i == 'coughc':
                        patient_vignette += f'The patient has cough with the following properties: {", ".join(pat_dat[0])}. '
                    elif len(pat_dat[0]) > 1 and i == 'nosec':
                        patient_vignette += f'The patient has nasal symptoms with the following properties: {", ".join(pat_dat[0])}. '
                    elif len(pat_dat[0]) > 1 and i == 'wild_contactc':
                        patient_vignette += f'The patient has had contact with wild animals including: {", ".join(pat_dat[0])}. '
                    else:
                        patient_vignette += f'The patients has {pat_dat[0].lower()}. '
                elif i == 'age_crf' and None not in pat_dat:
                    if len(pat_dat) > 1:
                        print("Expected only one value for 'age_crf', but got multiple.")
                        import pdb; pdb.set_trace()
                    patient_vignette += f'The patients age is {pat_dat[0]}. '
                elif i == 'eat_bush' and 1 in pat_dat:
                    patient_vignette += 'The patients has recently eaten bush meat. '
                elif i == 'funeral' and 1 in pat_dat:
                    patient_vignette += 'The patients has attended a funeral recently. '
                elif i == 'oxygen_saturation':
                    patient_vignette += f'The patients oxygen saturation is {pat_dat[0]}. '
                elif i == 'ph_urin':
                    patient_vignette += f'The patients urine pH is {pat_dat[0]}. '
                elif i == 'rodent_contact' and 1 in pat_dat:
                    patient_vignette += 'The patients has had contact with rodents. '
                elif i == 'spgr_urin':
                    patient_vignette += f'The patients urine specific gravity is {pat_dat[0]}. '
                elif i == 'urobilinogen_urin':
                    patient_vignette += f'The patients urine urobilinogen level is {pat_dat[0]}. '
                elif (i == 'insecticides' and 1 in pat_dat) or (i == 'insecticide' and 1 in pat_dat):
                    patient_vignette += 'The patients has been exposed to insecticides. '
                elif i == 'med1' and None not in pat_dat:
                    patient_vignette += f'The patients has taken {pat_dat[0]} medication. '
                elif i == 'med2' and None not in pat_dat:
                    patient_vignette += f'The patients has taken {pat_dat[0]} medication. '
                elif i == 'meds' and 1 in pat_dat:
                    patient_vignette += 'The patients has taken medications. '
                elif i == 'time1' and None not in pat_dat:
                    patient_vignette += f'The patient has taken their first medication for {pat_dat[0]}. '
                elif i == 'time2' and None not in pat_dat:
                    patient_vignette += f'The patient has taken their second medication for {pat_dat[0]}. '
                elif i == 'ethnicityc_crf' and None not in pat_dat:
                    if len(pat_dat) > 1:
                        print("Expected only one value for 'ethnicityc_crf', but got multiple.")
                    patient_vignette += f'The patients ethnicity is {pat_dat[0].lower()}. '
                elif i == 'other_occupation2' and None not in pat_dat:
                    if len(pat_dat) > 1:
                        print("Expected only one value for 'other_occupation2', but got multiple.")
                    patient_vignette += f'The patients occupation is {pat_dat[0].lower()}. '
                elif i == 'diagnosis_history' and None not in pat_dat:
                    patient_vignette += f'The patients has a history of diagnoses including: {", ".join(pat_dat[0])}. '
                elif i == 'othersymptoms_details' and None not in pat_dat:
                    patient_vignette += f'The patients has other symptoms including: {", ".join(pat_dat[0])}. '
                elif (i == 'med3' and None not in pat_dat) or (i == 'med4' and None not in pat_dat):
                    patient_vignette += f'The patients has taken {pat_dat[0]} medication. '
                elif None not in pat_dat and 0 not in pat_dat:
                    import pdb; pdb.set_trace()
        except Exception as e:
            print(f'Errored here with this error {e} for this {i}')
            import pdb; pdb.set_trace()
    
    return patient_vignette
