# Guia Git do grupo (Victor = dono do repositório e revisor)

Nome do repositório: `grupo05-deteccao-bordas-gradiente` (público).

## 1. Criar e publicar (Victor)
```bash
git add src/__init__.py gerar_figuras.py app notebooks imagens/entrada/teste.png imagens/saida/01_original.png
git commit -m "Integração: gerar_figuras, notebook, app Streamlit e imagem de teste"
# No GitHub: New repository > nome acima > Public > sem README (já existe um)
git remote set-url origin git@github.com:SEU_USUARIO/grupo05-deteccao-bordas-gradiente.git
git push -u origin main
```
Hoje o `origin` aponta para `FelipeAkryghti/atividade-final-CG`. Decidam qual repositório é o oficial e
atualizem `REPO_URL` na primeira célula do notebook e o badge do Colab no README.

## 2. Adicionar colaboradores
GitHub > Settings > Collaborators > Add people: Felipe, Fernando e Igor (permissão *Write*). Eles aceitam o convite por e-mail.

## 3. Proteger a main (recomendado)
Settings > Branches > Add rule em `main`: exigir Pull Request antes do merge. Só o Victor faz merge.

## 4. Cada integrante trabalha na sua branch
```bash
git clone git@github.com:SEU_USUARIO/grupo05-deteccao-bordas-gradiente.git
cd grupo05-deteccao-bordas-gradiente
git switch -c feat/utils             # Felipe  (Fernando: feat/operadores | Igor: feat/scharr)
# edita só os próprios arquivos
pytest testes/test_utils.py          # confere antes de subir
git add src/utils.py testes/test_utils.py
git commit -m "Implementa utils"
git push -u origin feat/utils
```
No GitHub: **Compare & pull request** > base `main` > descrever o que fez > Create.

## 5. Victor revisa e faz merge
```bash
git fetch origin
git switch feat/operadores && git pull
pytest                               # todos os testes
python gerar_figuras.py              # teste de integração
```
No GitHub: *Files changed* (comentar linhas) > *Approve* > **Merge pull request**. Depois: `git switch main && git pull`.
Ordem de merge: utils → operadores → scharr.

## 6. Atualizar a própria branch com a main
```bash
git switch feat/scharr
git fetch origin
git merge origin/main
```

## 7. Conflitos
Cada um edita arquivos diferentes, então conflito é raro. Se acontecer:
1. `git merge origin/main` avisa `CONFLICT in arquivo`; `git status` lista os arquivos.
2. Abra o arquivo e ache:
   ```
   <<<<<<< HEAD
   sua versão
   =======
   versão da main
   >>>>>>> origin/main
   ```
3. Deixe o texto final correto (pode misturar) e **apague as três linhas de marcação**.
4. `git add arquivo && git commit`, depois `git push`, e rode `pytest`. Para desistir: `git merge --abort`.

Regras: não editar arquivo de outro integrante (peça no PR ou no grupo). Se houver stub e código real, vale o código real do dono do módulo.

## 8. Antes da entrega
Todos os PRs mergeados, `git pull` na main, `pytest` verde, `python gerar_figuras.py`, `.gitignore` com `__pycache__/`,
notebook abrindo no Colab pelo badge e repositório **público** (testar em aba anônima).
