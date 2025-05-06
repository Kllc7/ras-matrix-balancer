import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import tempfile
import time
import os

# Cache the RAS function for better performance when parameters are reused
@st.cache_data
def ras(matrice_initiale, marges_lignes, marges_colonnes, tolerance=1e-6, max_iterations=1000):
    """
    Équilibre une matrice en utilisant la méthode RAS (Row and Column Scaling) optimisée.
    
    Args:
        matrice_initiale (numpy.ndarray): La matrice initiale à équilibrer.
        marges_lignes (numpy.ndarray): Les marges cibles pour les lignes.
        marges_colonnes (numpy.ndarray): Les marges cibles pour les colonnes.
        tolerance (float): La tolérance pour la convergence (défaut: 1e-6).
        max_iterations (int): Le nombre maximum d'itérations (défaut: 1000).
        
    Returns:
        tuple: (matrice équilibrée, nombre d'itérations, convergence atteinte, log d'erreur)
    """
    # Créer une copie pour éviter de modifier l'original
    matrice = np.copy(matrice_initiale).astype(np.float64)
    
    # Vérification des dimensions
    n_lignes, n_colonnes = matrice.shape
    if len(marges_lignes) != n_lignes or len(marges_colonnes) != n_colonnes:
        raise ValueError("Les dimensions des marges ne correspondent pas à celles de la matrice")
    
    # Vérifier si les sommes totales sont cohérentes
    somme_lignes = np.sum(marges_lignes)
    somme_colonnes = np.sum(marges_colonnes)
    if not np.isclose(somme_lignes, somme_colonnes, rtol=1e-5):
        st.warning(f"Attention: La somme des marges lignes ({somme_lignes}) est différente de la somme des marges colonnes ({somme_colonnes})")
    
    # Pour le suivi de la convergence
    log_erreurs = []
    convergence_atteinte = False
    
    # Éviter les divisions par zéro
    epsilon = np.finfo(float).eps
    
    for iteration in range(max_iterations):
        # Ajustement des lignes (vectorisé)
        totaux_lignes = np.sum(matrice, axis=1, keepdims=True)
        # Éviter division par zéro + vectorisation
        facteurs_lignes = np.where(totaux_lignes > epsilon, marges_lignes.reshape(-1, 1) / totaux_lignes, 0)
        matrice = matrice * facteurs_lignes
        
        # Ajustement des colonnes (vectorisé)
        totaux_colonnes = np.sum(matrice, axis=0, keepdims=True)
        # Éviter division par zéro + vectorisation
        facteurs_colonnes = np.where(totaux_colonnes > epsilon, marges_colonnes / totaux_colonnes, 0)
        matrice = matrice * facteurs_colonnes
        
        # Vérification de la convergence - calcul efficace de l'erreur
        ecart_lignes = np.sum(np.abs(np.sum(matrice, axis=1) - marges_lignes))
        ecart_colonnes = np.sum(np.abs(np.sum(matrice, axis=0) - marges_colonnes))
        ecart_total = ecart_lignes + ecart_colonnes
        log_erreurs.append(ecart_total)
        
        # Vérification de convergence
        if ecart_total < tolerance:
            convergence_atteinte = True
            break
            
        # Vérifier si l'erreur oscille ou stagne
        if iteration > 10 and abs(log_erreurs[-1] - log_erreurs[-2]) < tolerance / 100:
            if abs(log_erreurs[-1] - log_erreurs[-5]) < tolerance / 10:
                st.warning(f"Convergence ralentie détectée à l'itération {iteration+1}. L'erreur stagne à {ecart_total:.2e}")
                break
    
    return matrice, iteration + 1, convergence_atteinte, log_erreurs

def validate_excel_file(uploaded_file):
    """
    Valide le format du fichier Excel avant traitement.
    
    Returns:
        dict: Informations de validation avec statut et messages
    """
    result = {"valid": True, "messages": []}
    
    try:
        # Vérifier l'extension du fichier
        if not uploaded_file.name.endswith(('.xlsx', '.xls')):
            result["valid"] = False
            result["messages"].append("Le fichier doit être au format Excel (.xlsx ou .xls)")
            return result
            
        # Vérifier la présence des feuilles requises
        xls = pd.ExcelFile(uploaded_file)
        required_sheets = ['Matrice', 'Marges_Lignes', 'Marges_Colonnes']
        missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
        
        if missing_sheets:
            result["valid"] = False
            result["messages"].append(f"Feuilles manquantes: {', '.join(missing_sheets)}")
        
        return result
        
    except Exception as e:
        result["valid"] = False
        result["messages"].append(f"Erreur lors de la validation du fichier: {str(e)}")
        return result

def process_excel_file(uploaded_file):
    """
    Traite le fichier Excel téléchargé et extrait la matrice et les marges.
    """
    try:
        # Lecture des trois feuilles du fichier Excel
        xls = pd.ExcelFile(uploaded_file)
        
        # Lecture de la matrice
        df_matrice = pd.read_excel(xls, 'Matrice', index_col=0)
        matrice = df_matrice.values
        
        # Vérifier s'il y a des valeurs négatives ou NaN
        if np.any(np.isnan(matrice)):
            st.warning("La matrice contient des valeurs manquantes (NaN) qui seront traitées comme des zéros.")
            matrice = np.nan_to_num(matrice, nan=0.0)
            
        if np.any(matrice < 0):
            st.error("La matrice contient des valeurs négatives, ce qui n'est pas compatible avec la méthode RAS.")
            return None
        
        # Lecture des marges
        df_marges_lignes = pd.read_excel(xls, 'Marges_Lignes', index_col=0)
        marges_lignes = df_marges_lignes.values.flatten()
        
        df_marges_colonnes = pd.read_excel(xls, 'Marges_Colonnes', index_col=0)
        marges_colonnes = df_marges_colonnes.values.flatten()
        
        # Vérification des dimensions
        if len(marges_lignes) != matrice.shape[0] or len(marges_colonnes) != matrice.shape[1]:
            st.error(f"Incompatibilité des dimensions: Matrice {matrice.shape}, Marges lignes {len(marges_lignes)}, Marges colonnes {len(marges_colonnes)}")
            return None
            
        # Vérifier les marges nulles
        if np.any(marges_lignes <= 0) or np.any(marges_colonnes <= 0):
            st.warning("Certaines marges sont nulles ou négatives, ce qui peut causer des problèmes de convergence.")
        
        return {
            'matrice': matrice,
            'marges_lignes': marges_lignes,
            'marges_colonnes': marges_colonnes,
            'index': df_matrice.index,
            'columns': df_matrice.columns
        }
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Excel: {str(e)}")
        return None

def save_results_to_excel(matrice_equilibree, data, iterations, convergence, log_erreurs):
    """
    Sauvegarde les résultats dans un fichier Excel avec plusieurs feuilles.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"matrice_equilibree_{timestamp}.xlsx"
    
    # Utiliser un buffer en mémoire pour éviter l'écriture sur disque
    buffer = io.BytesIO()
    
    # Création d'un writer Excel
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Matrice équilibrée
        df_equilibree = pd.DataFrame(
            matrice_equilibree,
            index=data['index'],
            columns=data['columns']
        )
        df_equilibree.to_excel(writer, sheet_name='Matrice_Equilibree')
        
        # Informations sur la convergence
        info_conv = pd.DataFrame({
            'Métrique': ['Nombre d\'itérations', 'Convergence atteinte', 'Erreur finale'],
            'Valeur': [iterations, 'Oui' if convergence else 'Non', log_erreurs[-1] if log_erreurs else 'N/A']
        })
        info_conv.to_excel(writer, sheet_name='Info_Convergence', index=False)
        
        # Vérification des marges
        totaux_lignes = np.sum(matrice_equilibree, axis=1)
        totaux_colonnes = np.sum(matrice_equilibree, axis=0)
        
        verif_lignes = pd.DataFrame({
            'Index': data['index'],
            'Marge_Cible': data['marges_lignes'],
            'Marge_Obtenue': totaux_lignes,
            'Écart': np.abs(totaux_lignes - data['marges_lignes']),
            'Écart_%': np.where(data['marges_lignes'] > 0, 
                               100 * np.abs(totaux_lignes - data['marges_lignes']) / data['marges_lignes'],
                               0)
        })
        verif_lignes.to_excel(writer, sheet_name='Verification_Lignes')
        
        verif_colonnes = pd.DataFrame({
            'Colonne': data['columns'],
            'Marge_Cible': data['marges_colonnes'],
            'Marge_Obtenue': totaux_colonnes,
            'Écart': np.abs(totaux_colonnes - data['marges_colonnes']),
            'Écart_%': np.where(data['marges_colonnes'] > 0,
                               100 * np.abs(totaux_colonnes - data['marges_colonnes']) / data['marges_colonnes'],
                               0)
        })
        verif_colonnes.to_excel(writer, sheet_name='Verification_Colonnes')
        
        # Log des erreurs pour visualiser la convergence
        if log_erreurs:
            df_log = pd.DataFrame({
                'Iteration': list(range(1, len(log_erreurs) + 1)),
                'Erreur': log_erreurs
            })
            df_log.to_excel(writer, sheet_name='Log_Convergence', index=False)
    
    buffer.seek(0)
    return buffer, output_filename

def afficher_preview(data):
    """Affiche un aperçu des données chargées"""
    st.subheader("Aperçu des données")
    
    tab1, tab2, tab3 = st.tabs(["Matrice", "Marges Lignes", "Marges Colonnes"])
    
    with tab1:
        st.dataframe(pd.DataFrame(data['matrice'], index=data['index'], columns=data['columns']), height=250)
        st.text(f"Dimensions: {data['matrice'].shape}")
    
    with tab2:
        st.dataframe(pd.DataFrame({'Index': data['index'], 'Valeur': data['marges_lignes']}), height=250)
        st.text(f"Somme: {np.sum(data['marges_lignes'])}")
    
    with tab3:
        st.dataframe(pd.DataFrame({'Colonne': data['columns'], 'Valeur': data['marges_colonnes']}), height=250)
        st.text(f"Somme: {np.sum(data['marges_colonnes'])}")

def main():
    st.set_page_config(
        page_title="Équilibrage de Matrices - Méthode RAS",
        page_icon="📊",
        layout="wide",
    )
    
    st.title("📊 Équilibrage de Matrices - Méthode RAS")
    
    # Sidebar pour les instructions
    with st.sidebar:
        st.header("Instructions")
        st.write("""
        ### 1. Préparation du fichier Excel
        Votre fichier doit contenir 3 feuilles :
        - 'Matrice' : La matrice initiale à équilibrer
        - 'Marges_Lignes' : Les marges cibles pour les lignes
        - 'Marges_Colonnes' : Les marges cibles pour les colonnes
        
        ### 2. Format attendu
        - Dans la feuille 'Matrice', les indices doivent être en première colonne
        - Dans les feuilles 'Marges_Lignes' et 'Marges_Colonnes', les valeurs doivent être en deuxième colonne
        
        ### 3. Conseils pour la convergence
        - Vérifiez que la somme des marges lignes = somme des marges colonnes
        - Évitez les valeurs nulles dans les marges
        - Utilisez une tolérance adaptée à vos données
        """)
        
        # Exemple de téléchargement
        st.markdown("### Besoin d'un exemple ?")
        
        # Créer un exemple de fichier
        def create_example_file():
            # Créer un fichier Excel d'exemple
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Matrice
                indices = [f'Ligne_{i}' for i in range(1, 6)]
                colonnes = [f'Col_{i}' for i in range(1, 7)]
                matrice = np.random.rand(5, 6) * 100
                pd.DataFrame(matrice, index=indices, columns=colonnes).to_excel(writer, sheet_name='Matrice')
                
                # Marges lignes
                marges_lignes = np.random.rand(5) * 1000
                pd.DataFrame({'Index': indices, 'Valeur': marges_lignes}).to_excel(writer, sheet_name='Marges_Lignes')
                
                # Marges colonnes
                # Ajuster pour que la somme soit identique
                somme_lignes = np.sum(marges_lignes)
                marges_colonnes = np.random.rand(6)
                marges_colonnes = marges_colonnes * (somme_lignes / np.sum(marges_colonnes))
                pd.DataFrame({'Colonne': colonnes, 'Valeur': marges_colonnes}).to_excel(writer, sheet_name='Marges_Colonnes')
                
            buffer.seek(0)
            return buffer
        
        example_file = create_example_file()
        st.download_button(
            label="📥 Télécharger un fichier exemple",
            data=example_file,
            file_name="exemple_matrice_ras.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Zone principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choisissez votre fichier Excel", type=['xlsx', 'xls'])
    
    # Initialiser la session state si nécessaire
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'validated' not in st.session_state:
        st.session_state.validated = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if uploaded_file is not None:
        # Validation du fichier
        validation = validate_excel_file(uploaded_file)
        
        if not validation["valid"]:
            for msg in validation["messages"]:
                st.error(msg)
        else:
            # Traitement du fichier
            data = process_excel_file(uploaded_file)
            
            if data is not None:
                st.session_state.data = data
                st.session_state.validated = True
                
                # Afficher un aperçu des données
                afficher_preview(data)
                
                st.write("### Paramètres de l'algorithme")
                
                col1, col2 = st.columns(2)
                with col1:
                    tolerance_type = st.selectbox(
                        "Type de tolérance",
                        ["Standard (1e-6)", "Précise (1e-10)", "Très précise (1e-15)", "Personnalisée"]
                    )
                
                with col2:
                    if tolerance_type == "Personnalisée":
                        tolerance_exp = st.slider(
                            "Exposant de la tolérance (10^x)",
                            min_value=-20,
                            max_value=0,
                            value=-6,
                            help="Plus l'exposant est petit, plus la tolérance est précise"
                        )
                        tolerance = 10.0 ** tolerance_exp
                    else:
                        tolerance_map = {
                            "Standard (1e-6)": 1e-6,
                            "Précise (1e-10)": 1e-9,
                            "Très précise (1e-15)": 1e-12
                        }
                        tolerance = tolerance_map[tolerance_type]
                
                max_iterations = st.number_input(
                    "Nombre maximum d'itérations",
                    value=1000,
                    min_value=100,
                    max_value=10000000,
                    step=1000
                )
                
                if st.button("🚀 Lancer l'équilibrage", type="primary"):
                    with st.spinner("Équilibrage en cours..."):
                        # Créer une barre de progression
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Application de la méthode RAS
                        start_time = time.time()
                        matrice_equilibree, iterations, convergence, log_erreurs = ras(
                            data['matrice'],
                            data['marges_lignes'],
                            data['marges_colonnes'],
                            tolerance,
                            max_iterations
                        )
                        end_time = time.time()
                        
                        # Mise à jour de la barre de progression
                        progress_bar.progress(100)
                        status_text.text(f"Calcul terminé en {end_time - start_time:.2f} secondes")
                        
                        # Sauvegarde des résultats
                        buffer, output_filename = save_results_to_excel(matrice_equilibree, data, iterations, convergence, log_erreurs)
                        
                        # Stocker les résultats
                        st.session_state.results = {
                            'matrice': matrice_equilibree,
                            'iterations': iterations,
                            'convergence': convergence,
                            'log_erreurs': log_erreurs,
                            'buffer': buffer,
                            'filename': output_filename,
                            'time': end_time - start_time
                        }
                        
                        # Affichage des résultats
                        st.success(f"⏱️ Équilibrage terminé en {iterations} itérations ({end_time - start_time:.2f} secondes)")
                        if not convergence:
                            st.warning("⚠️ La convergence n'a pas été atteinte avec la tolérance spécifiée")
                        
                # Afficher les résultats s'ils existent
                if st.session_state.results:
                    # Afficher un résumé des résultats
                    st.subheader("Résultats de l'équilibrage")
                    
                    # Métriques de performance
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Itérations", st.session_state.results['iterations'])
                    with c2:
                        st.metric("Temps de calcul", f"{st.session_state.results['time']:.3f} s")
                    with c3:
                        st.metric("Convergence", "Atteinte" if st.session_state.results['convergence'] else "Non atteinte")
                    
                    # Visualisation de la convergence
                    if st.session_state.results['log_erreurs']:
                        st.subheader("Courbe de convergence")
                        df_log = pd.DataFrame({
                            'Iteration': list(range(1, len(st.session_state.results['log_erreurs']) + 1)),
                            'Erreur': st.session_state.results['log_erreurs']
                        })
                        
                        # Afficher la courbe de convergence avec échelle logarithmique
                        st.line_chart(df_log.set_index('Iteration'))
                        
                        # Afficher quelques échantillons de la matrice résultante
                        st.subheader("Aperçu de la matrice équilibrée")
                        result_df = pd.DataFrame(
                            st.session_state.results['matrice'],
                            index=data['index'],
                            columns=data['columns']
                        )
                        st.dataframe(result_df, height=250)
                    
                    # Bouton de téléchargement
                    st.download_button(
                        "📥 Télécharger les résultats complets",
                        st.session_state.results['buffer'],
                        file_name=st.session_state.results['filename'],
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

if __name__ == "__main__":
    main()