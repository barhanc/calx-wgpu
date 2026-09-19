import tsParser from '@typescript-eslint/parser';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import prettierPlugin from 'eslint-plugin-prettier';
import jsdocPlugin from 'eslint-plugin-jsdoc';

export default [
  {
    ignores: ['node_modules/', 'dist/', 'third-party/', '*.log', 'src/schema/'],
  },
  {
    files: ['**/*.ts', '**/*.tsx'],
    languageOptions: {
      parser: tsParser,
      ecmaVersion: 'latest',
      sourceType: 'module',
    },
    plugins: {
      '@typescript-eslint': tsPlugin,
      'prettier': prettierPlugin,
      'jsdoc': jsdocPlugin,
    },
    rules: {
      ...tsPlugin.configs.recommended.rules,
      ...jsdocPlugin.configs['recommended-typescript'].rules,
      'prettier/prettier': [
        'error',
        {
          printWidth: 100,
          quoteProps: 'consistent',
          singleQuote: true,
          tabWidth: 2,
          trailingComma: 'es5',
          useTabs: false,
        },
      ],
      'camelcase': 'error',
      'no-console': 'warn',
      'jsdoc/require-jsdoc': 'off',
      'jsdoc/require-param': ['error', { checkDestructured: false }],
      'jsdoc/check-param-names': ['error', { checkDestructured: false }],
      'jsdoc/require-yields-type': 'off',
      'jsdoc/require-yields-description': 'warn',
      'jsdoc/check-tag-names': ['error', { definedTags: ['category', 'property', 'internal'] }],
      'jsdoc/tag-lines': ['error', 'any'],
      'jsdoc/require-returns': 'off',
      'jsdoc/require-returns-description': 'off',
    },
    settings: {
      jsdoc: {
        tagNamePreference: {
          typeParam: 'typeParam',
        },
      },
    },
  },
];
